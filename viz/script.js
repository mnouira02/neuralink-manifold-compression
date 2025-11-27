import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

/* ================== Configuration ================== */
const COORDS_URL = 'viz_data/electrode_coords.json';
const EVENTS_URL = 'viz_data/source_events.json';

const TARGET_SCENE_DIAMETER = 3.0;
const ACTUAL_FS = 19531.0;
const RENDERED_SOURCE_SAMPLE = 200;

/* ================== State ================== */
let scene, camera, renderer, controls;
let electrodePoints = null;
let sourcePoints = null;

// AR Objects
let implantGroup = null;
let cortexMesh = null;
let connectivityLines = null;

let electrodeCoords = [];
let sourceEvents = [];

let sourceColorsAttribute = null;
let sourceSizesAttribute = null;
let renderedToEventIndex = [];
let sourceActivationTimers = [];

let currentTime = 0;
let totalTime = 0;
let lastTimestamp = 0;
let isPlaying = true;

// UI Parameters
let rotationSpeed = 0.002;
let sizeMultiplier = 2.5;
let activationTime = 0.6;
let currentSourceColor = new THREE.Color(0x00ffff);

// Feature Toggles
let showImplant = false;
let showCortex = false;
let showSynapses = false;

/* ================ Utility ================ */
async function loadJSON(url) {
    try {
        const r = await fetch(url);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return await r.json();
    } catch (err) {
        console.error('Failed to load JSON', url, err);
        return [];
    }
}

/* ================ Data Normalization ================ */
function normalizeAndScaleData(rawCoords, rawEvents) {
    const eventPositions = rawEvents.map(e => (e.position || [0, 0, 0]));
    const all = rawCoords.concat(eventPositions);
    if (!all.length) return { scaledCoords: [], scaledEvents: [] };

    const flat = new Float32Array(all.flat());
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

    for (let i = 0; i < flat.length; i += 3) {
        const x = flat[i], y = flat[i + 1], z = flat[i + 2];
        minX = Math.min(minX, x); maxX = Math.max(maxX, x);
        minY = Math.min(minY, y); maxY = Math.max(maxY, y);
        minZ = Math.min(minZ, z); maxZ = Math.max(maxZ, z);
    }

    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;
    const centerZ = (minZ + maxZ) / 2;
    const maxDim = Math.max(maxX - minX, maxY - minY, maxZ - minZ);
    const scale = TARGET_SCENE_DIAMETER / Math.max(maxDim, 1e-9);

    const scalePos = (p) => [(p[0] - centerX) * scale, (p[1] - centerY) * scale, (p[2] - centerZ) * scale];

    const scaledCoords = rawCoords.map(scalePos);
    const scaledEvents = rawEvents.map(e => Object.assign({}, e, { position: scalePos(e.position) }));

    return { scaledCoords, scaledEvents };
}

/* ================ Helper: Procedural Logo Texture ================ */
function createLogoTexture() {
    const size = 512;
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');

    // 1. Background (Transparent or matching cap color)
    // We leave it transparent so it layers over the white ceramic material
    ctx.clearRect(0, 0, size, size);

    // 2. Draw "NEURALINK" Text
    ctx.font = 'bold 60px Arial, sans-serif'; // Clean, modern font
    ctx.fillStyle = '#111111'; // Dark text for high contrast on white cap
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    // Draw text slightly offset to look like branding
    ctx.save();
    ctx.translate(size / 2, size / 2);
    ctx.rotate(-Math.PI / 2); // Rotate text to face "forward" relative to camera default
    ctx.fillText('NEURALINK', 0, 0);

    // Optional: Add a subtle "N" logo ring or dot if desired
    ctx.beginPath();
    ctx.arc(0, 0, 180, 0, Math.PI * 2);
    ctx.lineWidth = 4;
    ctx.strokeStyle = '#aaaaaa'; // Subtle ring
    ctx.stroke();
    ctx.restore();

    const texture = new THREE.CanvasTexture(canvas);
    texture.anisotropy = 4; // Sharper at angles
    return texture;
}

/* ================ AR / Overlay Generation (Realistic N1) ================ */

function createImplantModel() {
    const group = new THREE.Group();

    // --- 1. The N1 "Puck" (Multi-Material) ---
    // Real N1 is approx 23mm x 8mm. 
    // Scale: 1 unit ~ 1.33mm. 
    // Radius ~ 8.6 units. Height ~ 3.0 units (slightly thicker for visual presence)

    const radius = 8.6;
    const height = 2.5;
    const segments = 64; // High-res cylinder

    const coinGeo = new THREE.CylinderGeometry(radius, radius, height, segments);

    // Material 0: Side (Titanium Body) - Dark, metallic, rough
    const sideMat = new THREE.MeshStandardMaterial({
        color: 0x333333,
        metalness: 0.9,
        roughness: 0.4,
    });

    // Material 1: Top (Ceramic Cap) - White/Grey, glossy, with Logo
    const logoTex = createLogoTexture();
    const topMat = new THREE.MeshStandardMaterial({
        color: 0xeeeeee,      // Ceramic white
        map: logoTex,         // Apply generated logo
        metalness: 0.1,       // Ceramic is dielectric
        roughness: 0.2,       // Smooth/Glossy
        transparent: false
    });

    // Material 2: Bottom (Bio-interface) - Dark
    const bottomMat = new THREE.MeshStandardMaterial({ color: 0x111111 });

    // Apply materials array [Side, Top, Bottom]
    const coin = new THREE.Mesh(coinGeo, [sideMat, topMat, bottomMat]);

    // Position: "Flush with Skull" (High up)
    coin.position.set(0, 10.0, 0);

    // Rotate slightly so the text faces the camera nicely by default
    coin.rotation.y = Math.PI / 2;

    group.add(coin);

    // --- 2. The Threads (64 Count) ---
    // Threads hang from the bottom of the puck down into the brain.
    const threadMat = new THREE.LineBasicMaterial({
        color: 0x555555,
        transparent: true,
        opacity: 0.2
    });

    const numThreads = 64;

    for (let i = 0; i < numThreads; i++) {
        // Random insertion point within the active cortical zone (approx 3.0 units wide)
        const r = 2.0 * Math.sqrt(Math.random()); // Uniform disk distribution
        const theta = Math.random() * 2 * Math.PI;
        const x = r * Math.cos(theta);
        const z = r * Math.sin(theta);

        const points = [];
        // Start at bottom of Puck (10.0 - half height)
        points.push(new THREE.Vector3(x, 10.0 - (height / 2), z));
        // End deep in the cloud (Y=-3.0)
        points.push(new THREE.Vector3(x, -3.0, z));

        const geo = new THREE.BufferGeometry().setFromPoints(points);
        const line = new THREE.Line(geo, threadMat);
        group.add(line);
    }

    group.visible = false;
    return group;
}

function createCorticalSurface() {
    // This plane represents the "top" of the brain tissue.
    // It sits ABOVE the data cloud but BELOW the puck.
    const geo = new THREE.PlaneGeometry(8, 8); // Covers the whole recording area
    const mat = new THREE.MeshStandardMaterial({
        color: 0xffaa88,
        transparent: true,
        opacity: 0.15, // Visible but ghostly
        side: THREE.DoubleSide,
        depthWrite: false
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.rotation.x = -Math.PI / 2;

    // Y=2.0 is roughly "Top of the Cloud"
    // This visually separates the "Wiring" (above) from the "Neural Data" (below)
    mesh.position.set(0, 2.0, 0);

    mesh.visible = false;
    return mesh;
}

function initConnectivityLines(maxLines = 500) {
    // Pre-allocate a BufferGeometry for dynamic lines
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(maxLines * 2 * 3); // 2 points per line, 3 coords per point
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setDrawRange(0, 0); // Start with 0 lines

    const material = new THREE.LineBasicMaterial({
        color: 0xffffff,
        transparent: true,
        opacity: 0.4,
        blending: THREE.AdditiveBlending
    });

    const lines = new THREE.LineSegments(geometry, material);
    lines.frustumCulled = false;
    lines.visible = false;
    return lines;
}

/* ================ Scene Setup ================ */
function setupScene(scaledCoords, scaledEvents) {
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000814); // Dark blue-black
    scene.fog = new THREE.FogExp2(0x000814, 0.15); // Depth fog

    const w = window.innerWidth, h = window.innerHeight;
    camera = new THREE.PerspectiveCamera(60, w / h, 0.01, 100);
    camera.position.set(3, 2, 4);

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(w, h);
    document.body.appendChild(renderer.domElement);

    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 1.0;

    // Lights
    const ambi = new THREE.AmbientLight(0x404040, 2);
    scene.add(ambi);
    const dir = new THREE.DirectionalLight(0xffffff, 1);
    dir.position.set(5, 10, 7);
    scene.add(dir);

    // 1. Electrodes (Static Cloud)
    if (scaledCoords && scaledCoords.length > 0) {
        const flat = scaledCoords.flat();
        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.Float32BufferAttribute(flat, 3));
        const mat = new THREE.PointsMaterial({ color: 0x00cccc, size: 0.05 });
        electrodePoints = new THREE.Points(geo, mat);
        scene.add(electrodePoints);
    }

    // 2. Neurons (Source Events)
    if (scaledEvents && scaledEvents.length > 0) {
        const positions = [];
        const colors = [];
        const sizes = [];

        const stride = Math.max(1, Math.floor(scaledEvents.length / RENDERED_SOURCE_SAMPLE));
        renderedToEventIndex = [];
        sourceActivationTimers = [];

        for (let i = 0; i < scaledEvents.length; i += stride) {
            const p = scaledEvents[i].position;
            positions.push(p[0], p[1], p[2]);
            colors.push(0, 0, 0); // start black
            sizes.push(0.0);
            renderedToEventIndex.push(i);
            sourceActivationTimers.push(0);
        }

        const g = new THREE.BufferGeometry();
        g.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        sourceColorsAttribute = new THREE.Float32BufferAttribute(colors, 3);
        sourceSizesAttribute = new THREE.Float32BufferAttribute(sizes, 1);
        g.setAttribute('color', sourceColorsAttribute);
        g.setAttribute('size', sourceSizesAttribute);

        // Custom Shader for Neurons (Halo effect)
        const mat = new THREE.ShaderMaterial({
            uniforms: {
                baseScale: { value: 20.0 } // Global scaler
            },
            vertexShader: `
                attribute float size;
                attribute vec3 color;
                varying vec3 vColor;
                void main() {
                    vColor = color;
                    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
                    gl_PointSize = size * (300.0 / -mvPosition.z); // Perspective scaling
                    gl_Position = projectionMatrix * mvPosition;
                }
            `,
            fragmentShader: `
                varying vec3 vColor;
                void main() {
                    vec2 coord = gl_PointCoord - vec2(0.5);
                    float dist = length(coord);
                    if (dist > 0.5) discard;
                    // Glow gradient
                    float strength = 1.0 - (dist * 2.0);
                    strength = pow(strength, 1.5);
                    gl_FragColor = vec4(vColor, strength);
                }
            `,
            transparent: true,
            depthWrite: false,
            blending: THREE.AdditiveBlending
        });

        sourcePoints = new THREE.Points(g, mat);
        scene.add(sourcePoints);
    }

    // 3. AR Overlays
    implantGroup = createImplantModel();
    scene.add(implantGroup);

    cortexMesh = createCorticalSurface();
    scene.add(cortexMesh);

    connectivityLines = initConnectivityLines();
    scene.add(connectivityLines);
}

/* ================ Dynamic Updates ================ */

function updateConnectivity() {
    if (!showSynapses || !sourcePoints) {
        connectivityLines.visible = false;
        return;
    }
    connectivityLines.visible = true;

    // Identify active neurons
    const activeIndices = [];
    for (let i = 0; i < sourceActivationTimers.length; i++) {
        if (sourceActivationTimers[i] > (activationTime * 0.5)) { // Only if "freshly" active
            activeIndices.push(i);
        }
    }

    // Draw lines between simultaneously active neurons
    const positions = connectivityLines.geometry.attributes.position.array;
    let lineCount = 0;
    const maxLines = positions.length / 6;

    // Simple distance-based connectivity for visual effect
    // Real Hebbian logic would require spike history, but this simulates "firing together"
    for (let i = 0; i < activeIndices.length; i++) {
        for (let j = i + 1; j < activeIndices.length; j++) {
            if (lineCount >= maxLines) break;

            const idxA = activeIndices[i];
            const idxB = activeIndices[j];

            // Get positions from source geometry
            const ax = sourcePoints.geometry.attributes.position.getX(idxA);
            const ay = sourcePoints.geometry.attributes.position.getY(idxA);
            const az = sourcePoints.geometry.attributes.position.getZ(idxA);

            const bx = sourcePoints.geometry.attributes.position.getX(idxB);
            const by = sourcePoints.geometry.attributes.position.getY(idxB);
            const bz = sourcePoints.geometry.attributes.position.getZ(idxB);

            // Distance check (don't connect far-flung neurons)
            const dist = Math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2);
            if (dist < 1.0) {
                // Add line
                const baseIdx = lineCount * 6;
                positions[baseIdx] = ax; positions[baseIdx + 1] = ay; positions[baseIdx + 2] = az;
                positions[baseIdx + 3] = bx; positions[baseIdx + 4] = by; positions[baseIdx + 5] = bz;
                lineCount++;
            }
        }
    }

    connectivityLines.geometry.setDrawRange(0, lineCount * 2);
    connectivityLines.geometry.attributes.position.needsUpdate = true;
}

function updateState(dt) {
    if (!sourcePoints) return;

    for (let j = 0; j < renderedToEventIndex.length; j++) {
        const evtIdx = renderedToEventIndex[j];
        const evt = sourceEvents[evtIdx];
        const tEvent = (evt.t_sample || 0) / ACTUAL_FS;

        // Time distance
        const diff = currentTime - tEvent;
        let timer = sourceActivationTimers[j];

        // Trigger logic
        if (Math.abs(diff) < 0.01 && timer <= 0) {
            timer = activationTime;
            highlightNeuronList(j);
        }

        // Decay
        if (timer > 0) {
            timer -= dt;
            if (timer < 0) timer = 0;
        }
        sourceActivationTimers[j] = timer;

        // Visual mapping
        const intensity = timer > 0 ? (timer / activationTime) : 0;

        // Size: Base size + Pulse
        // sizeMultiplier comes from slider
        const sizeVal = (sizeMultiplier * 0.1) + (intensity * sizeMultiplier * 0.3);
        sourceSizesAttribute.setX(j, sizeVal);

        // Color: Mix between off-color and currentSourceColor
        if (intensity > 0) {
            sourceColorsAttribute.setXYZ(j,
                currentSourceColor.r * intensity,
                currentSourceColor.g * intensity,
                currentSourceColor.b * intensity
            );
        } else {
            sourceColorsAttribute.setXYZ(j, 0, 0, 0); // invisible when inactive
        }
    }

    sourceSizesAttribute.needsUpdate = true;
    sourceColorsAttribute.needsUpdate = true;
}

function highlightNeuronList(idx) {
    // Optional: Only update list every few frames to save DOM reflows if heavy
    const el = document.getElementById(`n-item-${idx}`);
    if (el) {
        el.classList.add('active');
        setTimeout(() => el.classList.remove('active'), 500);
    }
}

function buildNeuronList() {
    const list = document.getElementById('neuron-list');
    list.innerHTML = '';
    // Limit list size for performance
    const maxItems = Math.min(renderedToEventIndex.length, 50);
    for (let i = 0; i < maxItems; i++) {
        const div = document.createElement('div');
        div.className = 'neuron-item';
        div.id = `n-item-${i}`;
        div.innerText = `Neuron ID: ${renderedToEventIndex[i]}`;
        list.appendChild(div);
    }
}

/* ================ UI Listeners ================ */
function initUI() {
    // Colors
    document.getElementById('neuron-color').addEventListener('input', (e) => currentSourceColor.set(e.target.value));

    // Sliders
    document.getElementById('size-slider').addEventListener('input', (e) => sizeMultiplier = parseFloat(e.target.value));
    document.getElementById('rotation-slider').addEventListener('input', (e) => {
        rotationSpeed = parseFloat(e.target.value);
        controls.autoRotate = rotationSpeed > 0;
        controls.autoRotateSpeed = rotationSpeed * 1000;
    });
    document.getElementById('burst-slider').addEventListener('input', (e) => activationTime = parseFloat(e.target.value));

    // Toggles (AR)
    document.getElementById('toggle-implant').addEventListener('change', (e) => {
        showImplant = e.target.checked;
        if (implantGroup) implantGroup.visible = showImplant;
    });
    document.getElementById('toggle-cortex').addEventListener('change', (e) => {
        showCortex = e.target.checked;
        if (cortexMesh) cortexMesh.visible = showCortex;
    });
    document.getElementById('toggle-synapses').addEventListener('change', (e) => showSynapses = e.target.checked);

    // Playback
    const playBtn = document.getElementById('play-btn');
    playBtn.addEventListener('click', () => {
        isPlaying = !isPlaying;
        playBtn.innerText = isPlaying ? "Pause" : "Play";
    });

    // Timeline
    const timeline = document.getElementById('timeline');
    timeline.addEventListener('click', (e) => {
        const rect = timeline.getBoundingClientRect();
        const x = e.clientX - rect.left;
        currentTime = (x / rect.width) * totalTime;
        // clear timers
        sourceActivationTimers.fill(0);
    });
}

function updateTimelineUI() {
    const head = document.getElementById('time-head');
    const display = document.getElementById('current-time');
    const pct = totalTime > 0 ? (currentTime / totalTime) * 100 : 0;
    head.style.left = `${pct}%`;
    display.innerText = currentTime.toFixed(2);
}

/* ================ Loop ================ */
function animate(time) {
    requestAnimationFrame(animate);
    const dt = (time - lastTimestamp) / 1000;
    lastTimestamp = time;

    if (isPlaying && totalTime > 0) {
        currentTime += dt;
        if (currentTime > totalTime) currentTime = 0;
    }

    updateState(dt);
    updateConnectivity();
    updateTimelineUI();

    controls.update();
    renderer.render(scene, camera);
}

/* ================ Init ================ */
async function init() {
    initUI();
    const rawCoords = await loadJSON(COORDS_URL);
    const rawEvents = await loadJSON(EVENTS_URL);

    if (rawEvents.length) {
        const tSamples = rawEvents.map(e => e.t_sample || 0);
        const maxT = Math.max(...tSamples);
        totalTime = maxT / ACTUAL_FS;
        document.getElementById('total-time').innerText = totalTime.toFixed(2);
    }

    const { scaledCoords, scaledEvents } = normalizeAndScaleData(rawCoords, rawEvents);
    electrodeCoords = scaledCoords;
    sourceEvents = scaledEvents;

    setupScene(electrodeCoords, sourceEvents);
    buildNeuronList();

    animate(0);
}

init();