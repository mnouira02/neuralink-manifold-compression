# Blind Spatial Manifold Compression for Neural Interfaces

**A geometry-aware compression framework for high-density neural probes (Neuralink N1).**

## 🧠 The Concept
This project implements a blind calibration pipeline that recovers the **functional 3D geometry** of a neural electrode array solely from the correlation structure of the recorded noise and spikes. By mapping channels to a virtual 3D manifold, we can cluster spikes into **"Source Events"** (neurons) rather than channel events, achieving massive data compression.

![Compression Visualization](./viz/neuralink_logo.png)

## 🚀 Key Features
* **Blind Geometry Discovery:** Uses Multidimensional Scaling (MDS) to reconstruct electrode positions without physical specs.
* **Source-Based Compression:** Reduces redundant channel data into sparse $(t, x, y, z, amp)$ source packets.
* **Interactive 3D Dashboard:** A futuristic Three.js visualization featuring:
    * Physically accurate N1 Implant overlay (AR).
    * Dynamic synaptic connectivity lines.
    * Real-time playback of neural "ripples".

## 📊 Results
* **Compression Ratio:** ~466x (vs. 16-bit raw).
* **Information Preservation:** ~40% Spike Recall (F1 Score).

## 🛠️ Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline
Place your Neuralink challenge WAV files in data/raw_wavs/
```bash
python src/demo_pipeline.py
```
This will generate the geometry and events logs in viz/viz_data/

### 3. Launch the Visualization
(Pre-computed demo data is included, so you can skip Step 2!)

To view the 3D dashboard, simply serve the root directory:
```bash
python -m http.server 8000
```
Open http://localhost:8000/viz/ in your browser.

