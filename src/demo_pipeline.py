import os
import numpy as np
from typing import List, Tuple, Dict, Any
import wave
import struct
import json

# Core Dependencies
from scipy.signal import butter, filtfilt
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
from sklearn.cluster import DBSCAN

# Visualization Dependencies (Used for computation and internal logic)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Configuration ---
ACTUAL_FS = 19531.0
DATA_FOLDER_PATH = "./data/raw_wavs/"
EXPORT_FOLDER_PATH = "./viz_data/"
MAX_CHANNELS = 64
THRESHOLD_SIGMA = 3.0       # Relaxed detection threshold for recall
SPATIAL_RADIUS = 1.5        # DBSCAN radius in the MDS space
SPREAD_SIGMA = 1.0          # Spatial fall-off for reconstruction template
RECON_AMPLITUDE_SCALE = 5.0 # Template scaling factor for re-detection

###############################################################################
# 1. Data Loading & Basic Preprocessing
###############################################################################

class NeuralRecording:
    """Container for multi-channel neural time series."""
    def __init__(self, data: np.ndarray, fs: float, channel_ids: List[int] = None):
        assert data.ndim == 2, "Data must be (n_channels, n_samples)"
        self.data = data.astype(np.float32)
        self.fs = fs
        self.n_channels, self.n_samples = data.shape
        self.channel_ids = channel_ids or list(range(self.n_channels))

    @classmethod
    def from_wav_folder(cls, folder: str, fs: float, max_channels: int = None) -> "NeuralRecording":
        wav_files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".wav"))
        if max_channels is not None:
            wav_files = wav_files[:max_channels]
        
        if not wav_files:
            raise FileNotFoundError(f"No WAV files found in: {folder}")

        traces = []
        for f in wav_files:
            path = os.path.join(folder, f)
            try:
                with wave.open(path, 'rb') as wf:
                    if wf.getnchannels() != 1 or wf.getframerate() != int(fs):
                        continue
                    raw = wf.readframes(wf.getnframes())
                    x = np.frombuffer(raw, dtype=np.int16)
                    traces.append(x)
            except Exception:
                continue

        if not traces:
            raise ValueError(f"No compatible WAV files loaded with FS={fs}.")

        min_len = min(len(x) for x in traces)
        data = np.stack([x[:min_len] for x in traces], axis=0)
        print(f"Loaded {data.shape[0]} channels, {data.shape[1]} samples.")
        return cls(data=data, fs=fs)


def bandpass_filter(recording: NeuralRecording, f_lo: float = 300.0, f_hi: float = 3000.0, order: int = 3) -> NeuralRecording:
    """Basic bandpass filter for spike detection."""
    fs = recording.fs
    nyq = 0.5 * fs
    b, a = butter(order, [f_lo / nyq, f_hi / nyq], btype='band')
    filtered = filtfilt(b, a, recording.data, axis=1)
    return NeuralRecording(data=filtered, fs=fs, channel_ids=recording.channel_ids)


###############################################################################
# 2. Spike Detection & 3. Coincidence Matrix
###############################################################################

def detect_spikes(recording: NeuralRecording, threshold_sigma: float, refractory_ms: float = 1.0) -> Dict[int, np.ndarray]:
    """Simple threshold-based spike detector per channel."""
    data = recording.data
    fs = recording.fs
    n_channels, n_samples = data.shape
    spikes = {}
    refractory = int(refractory_ms * 1e-3 * fs)

    for ch in range(n_channels):
        x = data[ch]
        mad = np.median(np.abs(x - np.median(x)))
        sigma = 1.4826 * mad if mad > 0 else np.std(x)
        thr = -threshold_sigma * sigma
        idx = []
        i = 0
        while i < n_samples:
            if x[i] < thr:
                idx.append(i)
                i += refractory
            else:
                i += 1
        spikes[recording.channel_ids[ch]] = np.asarray(idx, dtype=np.int32)
    return spikes


def build_coincidence_matrix(spikes: Dict[int, np.ndarray], fs: float, window_ms: float = 0.2) -> np.ndarray:
    """Build a symmetric correlation matrix between channels."""
    ch_ids = sorted(spikes.keys())
    n_channels = len(ch_ids)
    C = np.zeros((n_channels, n_channels), dtype=np.float32)
    win = int(window_ms * 1e-3 * fs)

    for i, ch_i in enumerate(ch_ids):
        s_i = spikes[ch_i]
        if len(s_i) == 0: continue
        for j, ch_j in enumerate(ch_ids[i:], start=i):
            s_j = spikes[ch_j]
            if len(s_j) == 0: continue

            a, b, count = 0, 0, 0
            while a < len(s_i) and b < len(s_j):
                dt = s_i[a] - s_j[b]
                if abs(dt) <= win:
                    count += 1
                    a += 1
                    b += 1
                elif dt > 0: b += 1
                else: a += 1

            C[i, j] = C[j, i] = float(count)

    diag = np.diag(C).copy()
    for i in range(n_channels):
        for j in range(n_channels):
            denom = np.sqrt(diag[i] * diag[j])
            C[i, j] = C[i, j] / max(denom, 1e-6)

    C = 0.5 * (C + C.T)
    return C


def correlation_to_distance(C: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Convert correlation to distance (D_ij = 1 - C_ij) for MDS."""
    D = 1.0 - np.clip(C, 0.0, 1.0)
    np.fill_diagonal(D, 0.0)
    D = 0.5 * (D + D.T)
    return D


###############################################################################
# 4. Manifold Learning & 5. Source Event Extraction
###############################################################################

def recover_electrode_positions(D: np.ndarray, n_components: int = 3, random_state: int = 42) -> np.ndarray:
    """Use MDS to embed channels into a low-dimensional Euclidean space."""
    mds = MDS(n_components=n_components, dissimilarity="precomputed", random_state=random_state, n_init=1, max_iter=3000, eps=1e-9, normalized_stress=False)
    coords = mds.fit_transform(D)
    pca = PCA(n_components=n_components)
    coords_aligned = pca.fit_transform(coords)
    return coords_aligned


def extract_source_events(recording: NeuralRecording, spikes: Dict[int, np.ndarray], coords: np.ndarray, spatial_radius: float, temporal_window_ms: float, min_cluster_size: int = 2) -> List[Dict[str, Any]]:
    """Convert per-channel spikes into amplitude-weighted source events."""
    fs = recording.fs
    temporal_window_samples = int(temporal_window_ms * 1e-3 * fs)
    ch_ids = sorted(spikes.keys())
    id_to_idx = {ch_id: i for i, ch_id in enumerate(ch_ids)}

    events = []
    for ch_id in ch_ids:
        ch_idx = id_to_idx[ch_id]
        t_indices = spikes[ch_id]
        for t in t_indices:
            w = 5
            t0 = max(0, t - w)
            t1 = min(recording.n_samples, t + w)
            segment = recording.data[ch_idx, t0:t1]
            amp = float(np.min(segment))
            events.append((t, ch_id, ch_idx, amp))

    if not events: return []
    events.sort(key=lambda e: e[0])

    source_events = []
    i = 0
    while i < len(events):
        t_ref = events[i][0]
        
        j = i
        cluster_candidates = []
        while j < len(events) and events[j][0] - t_ref <= temporal_window_samples:
            cluster_candidates.append(events[j])
            j += 1

        if len(cluster_candidates) < min_cluster_size:
            i = j
            continue

        idxs = [e[2] for e in cluster_candidates]
        positions = coords[idxs]

        clustering = DBSCAN(eps=spatial_radius, min_samples=min_cluster_size).fit(positions)
        labels = clustering.labels_

        for lbl in set(labels):
            if lbl == -1: continue
            
            members = [
                (t, ch_id, ch_idx, amp, pos)
                for (t, ch_id, ch_idx, amp), pos, lab
                in zip(cluster_candidates, positions, labels)
                if lab == lbl
            ]
            if len(members) < min_cluster_size: continue

            amps = np.array([abs(m[3]) for m in members])
            pos_arr = np.stack([m[4] for m in members], axis=0)
            t_arr = np.array([m[0] for m in members])

            w = amps / np.maximum(amps.sum(), 1e-6)
            centroid_pos = (pos_arr * w[:, None]).sum(axis=0)
            centroid_t = int((t_arr * w).sum())

            source_event = {
                "t_sample": centroid_t,
                "position": centroid_pos,
                "amplitude": float(amps.mean()),
            }
            source_events.append(source_event)

        i = j 
    source_events.sort(key=lambda e: e["t_sample"])
    return source_events


###############################################################################
# 6. Compression Format & Encoding
###############################################################################

class SourceEventCodec:
    """Conceptual binary encoder/decoder for source events (Fixed-bit proxy)."""
    def __init__(self, fs: float, position_scale: float = 10.0, max_amp: float = 1024.0):
        self.fs = fs
        self.position_scale = position_scale
        self.max_amp = max_amp

    def quantize_position(self, pos: np.ndarray) -> Tuple[int, int, int]:
        scaled = pos * self.position_scale
        q = np.clip(np.round(scaled), -32768, 32767).astype(np.int16) 
        return tuple(int(v) for v in q)

    def dequantize_position(self, qx: int, qy: int, qz: int) -> np.ndarray:
        arr = np.array([qx, qy, qz], dtype=np.float32)
        return arr / self.position_scale

    def quantize_amplitude(self, amp: float) -> int:
        q_amp = np.clip(abs(amp), 0, self.max_amp)
        return int(np.round(q_amp * (1023 / self.max_amp)))

    def decode_amplitude(self, q_amp: int) -> float:
        return float(q_amp) * (self.max_amp / 1023.0)

    def encode_events(self, events: List[Dict[str, Any]]) -> bytes:
        out = bytearray()
        prev_t = 0
        for ev in events:
            dt = ev["t_sample"] - prev_t
            prev_t = ev["t_sample"]
            qx, qy, qz = self.quantize_position(ev["position"])
            q_amp = self.quantize_amplitude(ev["amplitude"])

            out += struct.pack("<IhhhH", int(dt), qx, qy, qz, q_amp)
        return bytes(out)

    def decode_events(self, blob: bytes) -> List[Dict[str, Any]]:
        events = []
        offset = 0
        prev_t = 0
        rec_size = 12
        while offset + rec_size <= len(blob):
            try:
                dt, qx, qy, qz, q_amp = struct.unpack_from("<IhhhH", blob, offset)
                offset += rec_size
            except struct.error:
                break

            t = prev_t + dt
            prev_t = t
            pos = self.dequantize_position(qx, qy, qz)
            amp = self.decode_amplitude(q_amp)
            
            events.append({
                "t_sample": t,
                "position": pos,
                "amplitude": amp,
            })
        return events


###############################################################################
# 7. Reconstruction & Evaluation Utilities
###############################################################################

def extract_average_template(recording_bp: NeuralRecording, spikes: Dict[int, np.ndarray], template_ms: float = 2.0) -> np.ndarray:
    """Extracts the normalized average waveform (template) across all channels/spikes."""
    fs = recording_bp.fs
    half_len = int(template_ms * 1e-3 * fs / 2)
    template_len = 2 * half_len
    all_templates = []
    
    for ch_idx in range(recording_bp.n_channels):
        ch_id = recording_bp.channel_ids[ch_idx]
        spike_times = spikes.get(ch_id, np.array([], dtype=np.int32))
        
        for t in spike_times:
            start = t - half_len
            end = t + half_len
            
            if start < 0 or end > recording_bp.n_samples: continue
            
            segment = recording_bp.data[ch_idx, start:end]
            if segment.shape[0] == template_len:
                all_templates.append(segment)

    if not all_templates:
        print("Warning: Template extraction failed. Returning zero template.")
        return np.zeros(template_len, dtype=np.float32)
        
    avg_template = np.mean(np.stack(all_templates, axis=0), axis=0)
    
    min_val = avg_template.min()
    if abs(min_val) > 1e-6:
        avg_template /= abs(min_val)
    
    return avg_template


def reconstruct_channels_from_sources(recording: NeuralRecording, coords: np.ndarray, source_events: List[Dict[str, Any]], spike_template: np.ndarray, spread_sigma: float = 1.0) -> np.ndarray:
    """Synthesize an approximate multi-channel recording from source events."""
    n_channels, n_samples = recording.n_channels, recording.n_samples
    recon = np.zeros((n_channels, n_samples), dtype=np.float32)
    half_len = len(spike_template) // 2

    for ev in source_events:
        t0 = ev["t_sample"]
        pos = ev["position"]
        amp = ev["amplitude"]
        
        dists = np.linalg.norm(coords - pos[None, :], axis=1)
        gains = np.exp(-0.5 * (dists / spread_sigma) ** 2)
        gains /= gains.max() + 1e-6 

        start = max(0, t0 - half_len)
        end = min(n_samples, t0 - half_len + len(spike_template))
        templ_len = end - start
        
        if templ_len > 0:
            segment = spike_template[:templ_len]

            for ch in range(n_channels):
                recon[ch, start:end] += gains[ch] * amp * segment

    return recon

def compute_spike_timing_agreement(original_spikes: Dict[int, np.ndarray], reconstructed_recording: NeuralRecording) -> Dict[str, float]:
    """Compare original spike times to spikes re-detected on reconstructed data."""
    rec_spikes = detect_spikes(reconstructed_recording, threshold_sigma=THRESHOLD_SIGMA)
    fs = reconstructed_recording.fs
    tol_samples = int(0.2 * 1e-3 * fs)

    tp, fp, fn = 0, 0, 0

    for ch_id, orig_ts in original_spikes.items():
        rec_ts = rec_spikes.get(ch_id, np.array([], dtype=np.int32))
        used = np.zeros(len(rec_ts), dtype=bool)

        for t in orig_ts:
            if len(rec_ts) == 0: fn += 1; continue
                
            diffs = np.abs(rec_ts - t)
            j = np.argmin(diffs)
            
            if diffs[j] <= tol_samples and not used[j]:
                tp += 1
                used[j] = True
            else:
                fn += 1
        fp += (~used).sum()

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


###############################################################################
# 8. Data Export and Visualization
###############################################################################

def export_data_for_web(coords: np.ndarray, decoded_events: List[Dict[str, Any]], export_path: str):
    """
    Saves MDS coordinates and source event centroids to JSON files.
    """
    os.makedirs(export_path, exist_ok=True)
    
    # 1. Export MDS Coordinates (The Static Grid)
    coords_list = coords.tolist()
    with open(os.path.join(export_path, 'electrode_coords.json'), 'w') as f:
        json.dump(coords_list, f, indent=2)
    print(f"   -> Exported {len(coords_list)} electrode coordinates to JSON.")

    # 2. Export Source Events
    exportable_events = []
    for ev in decoded_events:
        exportable_events.append({
            "t_sample": ev["t_sample"],
            "position": ev["position"].tolist(),
            "amplitude": ev["amplitude"]
        })

    with open(os.path.join(export_path, 'source_events.json'), 'w') as f:
        json.dump(exportable_events, f, indent=2)
    print(f"   -> Exported {len(exportable_events)} source events to JSON.")


def visualize_sources_3d(coords: np.ndarray, source_events: List[Dict[str, Any]]):
    """Matplotlib visualization (for debugging geometry before Three.js)."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c='gray', marker='o', label='Virtual Electrodes', alpha=0.5, s=30)
    
    source_positions = np.array([ev['position'] for ev in source_events])
    if len(source_positions) == 0:
        print("No source events to visualize.")
        plt.show()
        return

    dbscan_sources = DBSCAN(eps=0.1, min_samples=5).fit(source_positions) 
    labels = dbscan_sources.labels_
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    print(f"   -> Found {len(unique_labels) - (1 if -1 in unique_labels else 0)} unique neuron centers for visualization.")
    
    for k, col in zip(unique_labels, colors):
        if k == -1: continue 

        class_member_mask = (labels == k)
        xy = source_positions[class_member_mask]
        centroid = np.mean(xy, axis=0)
        
        ax.scatter(centroid[0], centroid[1], centroid[2], 
                   marker='X', s=300, c=[col], label=f'Neuron Source {k}',
                   edgecolor='k', linewidth=2)

    ax.set_title("Virtual 3D Map: Electrodes vs. Discovered Neuron Sources (Matplotlib)")
    ax.set_xlabel('MDS Dim 1')
    ax.set_ylabel('MDS Dim 2')
    ax.set_zlabel('MDS Dim 3')
    ax.legend(loc='best', fontsize='medium')
    plt.show()


###############################################################################
# 9. Main Pipeline Execution
###############################################################################

def demo_pipeline(data_folder: str, export_folder: str, fs: float, max_channels: int) -> None:
    
    # ----------------------------------------------------
    # CALIBRATION MODE (Geometry Discovery)
    # ----------------------------------------------------
    print("\n--- CALIBRATION MODE: Discovering Geometry ---")
    
    recording_raw = NeuralRecording.from_wav_folder(folder=data_folder, fs=fs, max_channels=max_channels)
    recording_bp = bandpass_filter(recording_raw)
    spikes = detect_spikes(recording_bp, threshold_sigma=THRESHOLD_SIGMA)
    total_spikes = sum(len(s) for s in spikes.values())
    print(f"   -> Total detected spikes (Sigma={THRESHOLD_SIGMA}): {total_spikes}")
    
    print("3. Building Coincidence Matrix...")
    C = build_coincidence_matrix(spikes, fs=fs)
    D = correlation_to_distance(C)
    
    print("4. Recovering Virtual 3D Electrode Map (MDS)...")
    coords = recover_electrode_positions(D, n_components=3)
    
    # ----------------------------------------------------
    # COMPRESSION & VALIDATION
    # ----------------------------------------------------
    print("\n--- COMPRESSION & VALIDATION ---")
    
    print(f"6. Extracting Source Events (Radius={SPATIAL_RADIUS}, MinSize=2)...")
    source_events = extract_source_events(
        recording=recording_bp, spikes=spikes, coords=coords,
        spatial_radius=SPATIAL_RADIUS, temporal_window_ms=0.2, min_cluster_size=2
    )
    print(f"   -> Total Source Events extracted: {len(source_events)}")
    
    # 7. Encode & Decode (Measures compression size)
    codec = SourceEventCodec(fs=fs, position_scale=10.0)
    blob = codec.encode_events(source_events)
    decoded_events = codec.decode_events(blob)
    
    raw_data_size_bits = recording_raw.data.size * 16
    compressed_size_bits = len(blob) * 8
    compression_ratio = raw_data_size_bits / compressed_size_bits
    
    print(f"   -> Compressed blob size: {len(blob) / 1024:.2f} KB")
    print(f"   -> **Compression Ratio (Size): {compression_ratio:.2f}x**")
    
    # 8. Reconstruction & F1 Score
    print("\n--- INFORMATION PRESERVATION ---")
    spike_template_normalized = extract_average_template(recording_bp, spikes, template_ms=2.0)
    scaled_template = spike_template_normalized * RECON_AMPLITUDE_SCALE
    
    recon = reconstruct_channels_from_sources(
        recording=recording_raw, coords=coords, source_events=decoded_events,
        spike_template=scaled_template, spread_sigma=SPREAD_SIGMA
    )
    recording_recon = NeuralRecording(data=recon, fs=fs)
    metrics = compute_spike_timing_agreement(original_spikes=spikes, reconstructed_recording=recording_recon)
    
    print("\n--- Final Metrics ---")
    print(f"Spike Timing F1 Score: {metrics['f1']:.4f}")
    print(f"Recall (Spikes Recovered): {metrics['recall']:.4f}")
    
    # 9. Data Export for Three.js
    print("\n--- EXPORTING FOR THREE.JS VISUALIZATION ---")
    export_data_for_web(coords, decoded_events, export_folder)
    
if __name__ == "__main__":
    
    # Check for output directory existence
    if not os.path.isdir(EXPORT_FOLDER_PATH):
        print(f"Creating export directory: {EXPORT_FOLDER_PATH}")
        os.makedirs(EXPORT_FOLDER_PATH)

    try:
        demo_pipeline(data_folder=DATA_FOLDER_PATH, export_folder=EXPORT_FOLDER_PATH, fs=ACTUAL_FS, max_channels=MAX_CHANNELS)
    except (FileNotFoundError, ValueError) as e:
        print(f"\n--- FATAL ERROR ---")
        print(f"Data loading failed: {e}.")
    except Exception as e:
        print(f"\n--- EXECUTION ERROR ---")
        print(f"An unexpected error occurred: {e}")