# Blind Spatial Manifold Compression for Neural Interfaces

## 🌟 [View the Live 3D Visualization](https://mnouira02.github.io/neuralink-manifold-compression/)

### 📖 [Read the Story on Medium](https://medium.com/@mnouira/i-broke-the-rules-to-solve-the-neuralink-challenge-88949bccbf45)

## 🧠 The Concept
This project implements a blind calibration pipeline that recovers the **functional 3D geometry** of a neural electrode array solely from the correlation structure of the recorded noise and spikes. By mapping channels to a virtual 3D manifold, we can cluster spikes into **"Source Events"** (neurons) rather than channel events, achieving massive data compression.

## ⚠️ The Data Paradox 
The Neuralink Challenge provided ~143MB of data, which represents roughly one hour of single-channel recording (or non-simultaneous segments). A real N1 implant generates ~90GB of simultaneous data per hour across 1,024 channels.

To test my "Blind Spatial Manifold" hypothesis, I had to simulate the N1's architecture.

I treated the provided sequential file segments as if they were simultaneous spatial channels. While this means the recovered geometry in this demo is a simulation based on the dataset's noise properties, the algorithm pipeline is exactly what would run on the physical chip to recover the real electrode positions.

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

