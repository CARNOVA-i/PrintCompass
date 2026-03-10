import "./style.css";
import * as THREE from "three";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { preprocessGeometry } from "./analyze/preprocess.js";
import { buildCandidates } from "./analyze/candidates.js";
import { scoreOverhang } from "./analyze/scoreOverhang.js";
import { scoreStrength } from "./analyze/scoreStrength.js";
import { STLExporter } from "three/examples/jsm/exporters/STLExporter.js";
import { computeInspectorReport } from "./analyze/inspector.js";



let lastSTLBuffer = null;
let lastAnalysisData = null;

let baseGeometry = null;   // pristine, never rotated
let viewMesh = null;       // what the user sees

let bestQuatStrength = null;
let bestQuatSupports = null;

let loadArrow = null;      // your arrow helper
let currentViewQuat = new THREE.Quaternion(); // tracks what is displayed

let lastFocusEl = null;

let viewMode = "strength"; // "strength" | "supports"
let rankedCandidatesStrength = [];
let rankedCandidatesSupports = [];
let currentCandidateIndex = 0;


function openHelp() {
  lastFocusEl = document.activeElement;

  helpModal.removeAttribute("hidden");
  helpModal.classList.add("open");    // pick one system
  helpModal.classList.remove("show"); // optional cleanup
  helpModal.setAttribute("aria-hidden", "false");

  helpClose?.focus(); // or helpOk?.focus()
}


function closeHelp() {
  // Pick a safe focus target outside the modal
  const fallback = document.getElementById("helpBtn");

  // If current focus is inside the modal, move it out first
  if (helpModal && helpModal.contains(document.activeElement)) {
    (lastFocusEl && !helpModal.contains(lastFocusEl) ? lastFocusEl : fallback)?.focus();
  } else {
    lastFocusEl?.focus();
  }

  // Now hide
  helpModal.classList.remove("open");
  helpModal.classList.remove("show"); // in case you still have the other system
  helpModal.setAttribute("aria-hidden", "true");
  helpModal.setAttribute("hidden", "");

  if (dontShowAgain?.checked) {
    localStorage.setItem("spoa_hide_help", "1");
  }
}



// Cura/export space is Z-up. Viewer space is Y-up.
// This converts a Z-up orientation into your Y-up viewer.
const qZupToYup = new THREE.Quaternion().setFromEuler(
  new THREE.Euler(-Math.PI / 2, 0, 0) // rotate -90° about X
);



const qYupToZup = qZupToYup.clone().invert();

let hasManualViewRotation = false;

function getViewerQuaternion(zUpQuat) {
  return qZupToYup.clone().multiply(zUpQuat);
}

function getCurrentAnalysisQuaternion() {
  return viewMode === "supports" ? bestQuatSupports : bestQuatStrength;
}

function getLoadAxisSelectorValue() {
  const el =
    document.getElementById("loadDir") ||
    document.getElementById("loadAxis") ||
    document.querySelector("[name='loadDir']") ||
    document.querySelector("[name='loadAxis']");

  return el?.value ?? "principal";
}

function normalizeAxisSelection(value) {
  switch ((value ?? "").trim().toLowerCase()) {
    case "x":
    case "+x":
      return "+x";
    case "-x":
      return "-x";
    case "y":
    case "+y":
      return "+y";
    case "-y":
      return "-y";
    case "z":
    case "+z":
      return "+z";
    case "-z":
      return "-z";
    case "principalaxis":
    case "principal-axis":
    case "auto":
    case "principal":
    default:
      return "principal";
  }
}

function getExportMode() {
  const raw = document.getElementById("exportMode")?.value ?? viewMode;

  switch ((raw ?? "").trim().toLowerCase()) {
    case "strength":
      return "strength";
    case "support":
    case "supports":
    case "overhang":
      return "supports";
    case "shown":
    case "current":
    case "view":
      return "view";
    default:
      return viewMode;
  }
}

function getActiveRankedCandidates() {
  return viewMode === "supports" ? rankedCandidatesSupports : rankedCandidatesStrength;
}

function formatCandidateScore(item) {
  if (!item) return "STR - | SUP -";
  return `STR ${Math.round(item.strengthScore)} | SUP ${Math.round(item.supportsScore)}`;
}

function updateCandidateUI() {
  const list = getActiveRankedCandidates();
  const item = list[currentCandidateIndex] ?? null;
  const prevBtn = document.getElementById("candidatePrevBtn");
  const nextBtn = document.getElementById("candidateNextBtn");
  const indexLabel = document.getElementById("candidateIndexLabel");
  const scoreLabel = document.getElementById("candidateScoreLabel");

  if (indexLabel) {
    indexLabel.textContent = list.length ? `Cand ${currentCandidateIndex + 1}/${list.length}` : "Cand -/-";
  }

  const cand = list[currentCandidateIndex] ?? null;
  
  if (scoreLabel) {
    scoreLabel.textContent = cand
      ? `STR ${cand.strengthScore?.toFixed(1)} | SUP ${cand.supportsScore?.toFixed(0)}`
      : "STR - | SUP -";
  }

  if (prevBtn) prevBtn.disabled = list.length === 0 || currentCandidateIndex <= 0;
  if (nextBtn) nextBtn.disabled = list.length === 0 || currentCandidateIndex >= list.length - 1;
}

function showCandidate(index) {
  const list = getActiveRankedCandidates();
  if (!list.length) {
    currentCandidateIndex = 0;
    updateCandidateUI();
    return;
  }

  currentCandidateIndex = Math.max(0, Math.min(index, list.length - 1));
  applyViewQuaternion(list[currentCandidateIndex].quaternion);
  updateCandidateUI();
}

function rebuildRankedCandidates(strengthScores, supportsScores) {
  const strengthByIdx = new Map(strengthScores.map((item) => [item.idx, item.score]));
  const supportsByIdx = new Map(supportsScores.map((item) => [item.idx, item.score]));

  const buildRankedList = (sorted) =>
    sorted.slice(0, 5).map((item, rank) => ({
      rank: rank + 1,
      quaternion: item.quat?.clone?.() || null,
      strengthScore: strengthByIdx.get(item.idx),
      supportsScore: supportsByIdx.get(item.idx),
    }));

  rankedCandidatesStrength = buildRankedList(strengthScores);
  rankedCandidatesSupports = buildRankedList(supportsScores);
  currentCandidateIndex = 0;
  updateCandidateUI();
}

function renderInspectorHUD(rep) {
  const hud = document.getElementById("inspectorHud");
  const pills = document.getElementById("inspectorPills");
  const line1 = document.getElementById("inspectorLine1");
  const pre = document.getElementById("inspectorJson");
  const body = document.getElementById("inspectorHudBody");
  const btnCollapse = document.getElementById("inspectorToggleBtn");

  const sum = document.getElementById("inspectorSummary");
  const warns = document.getElementById("inspectorWarnings");
  const rawBtn = document.getElementById("inspectorRawBtn");


  hud.hidden = false;

  if (!hud || !pills || !line1 || !pre || !sum || !warns) return;

  

  // Collapse HUD body (existing behavior)
  if (btnCollapse && !btnCollapse._wired) {
    btnCollapse._wired = true;
    btnCollapse.addEventListener("click", () => {
      const collapsed = body.style.display !== "none";
      body.style.display = collapsed ? "none" : "block";
      btnCollapse.textContent = collapsed ? "+" : "–";
      btnCollapse.title = collapsed ? "Expand" : "Collapse";
    });
  }

  // Raw toggle (NEW)
  if (rawBtn && !rawBtn._wired) {
    rawBtn._wired = true;
    rawBtn.addEventListener("click", () => {
      const isHidden = pre.style.display === "none";
      pre.style.display = isHidden ? "block" : "none";
      rawBtn.textContent = isHidden ? "Hide" : "Show";
    });
  }

  const scoreClass = rep.score >= 85 ? "ok" : rep.score >= 60 ? "warn" : "bad";
  const wtClass = rep.edgeStats.watertight ? "ok" : "bad";

  pills.innerHTML = `
    <span class="inspectorPill ${scoreClass}">Score ${rep.score}</span>
    <span class="inspectorPill ${wtClass}">${rep.edgeStats.watertight ? "Watertight" : "Open"}</span>
    <span class="inspectorPill">Shells ${rep.shells}</span>
  `;

  line1.textContent =
    `Tris ${rep.triangles} | Size ${rep.size?.x?.toFixed(2)} x ${rep.size?.y?.toFixed(2)} x ${rep.size?.z?.toFixed(2)} | Units ${rep.unitGuess}`;

  // Summary (human)
  const sa = Number.isFinite(rep.surfaceArea) ? rep.surfaceArea.toFixed(1) : "-";
  const vol = Number.isFinite(rep.volume) ? rep.volume.toFixed(1) : "-";
  sum.innerHTML = `
    <b>Summary</b><br>
    Surface area: ${sa}<br>
    Volume: ${vol}<br>
    Degenerate triangles: ${rep.degenerateTriangles}
  `;

  // Warnings list (human)
  warns.innerHTML = "";
  const addW = (level, title, detail) => {
    const d = document.createElement("div");
    d.className = `inspectorWarn ${level}`;
    d.innerHTML = `<b>${title}</b><div>${detail}</div>`;
    warns.appendChild(d);
  };

  if (rep.edgeStats.watertight) {
    addW("ok", "Mesh health", "Watertight, no boundary edges, no non-manifold edges detected.");
  } else {
    if (rep.edgeStats.boundaryEdges > 0) addW("bad", "Open boundaries", `${rep.edgeStats.boundaryEdges} boundary edges found. Likely holes.`);
    if (rep.edgeStats.nonManifoldEdges > 0) addW("bad", "Non-manifold edges", `${rep.edgeStats.nonManifoldEdges} edges used by more than 2 triangles.`);
  }

  if (rep.shells > 1) addW("warn", "Multiple shells", `${rep.shells} shells detected. Could be intentional, could be floating junk.`);
  if (rep.degenerateTriangles > 0) addW("warn", "Degenerate triangles", `${rep.degenerateTriangles} zero-area triangles found.`);
  if (rep.unitGuess !== "mm (likely)") addW("warn", "Units", `Units are a heuristic guess: ${rep.unitGuess}. Verify scale before printing.`);

  // Raw JSON (hidden by default, user can show)
  pre.textContent = JSON.stringify(rep, null, 2);
}


const toggleViewBtn = document.getElementById("toggleViewBtn");

toggleViewBtn?.addEventListener("click", () => {
  console.log("toggleViewBtn clicked");
});





function applyViewQuaternion(q) {
  if (!viewMesh) return;

  hasManualViewRotation = false;

  // Convert from Z-up (solver/export) to Y-up (viewer)
  currentViewQuat.copy(getViewerQuaternion(q));

  // Reset transforms
  viewMesh.position.set(0, 0, 0);
  viewMesh.rotation.set(0, 0, 0);

  // Apply rotation
  viewMesh.setRotationFromQuaternion(currentViewQuat);
  reflowViewMesh();
}

function reflowViewMesh() {
  if (!viewMesh) return;

  viewMesh.updateMatrixWorld(true);

  // Ground to build plate (Y up)
  const box = new THREE.Box3().setFromObject(viewMesh);
  viewMesh.position.y -= box.min.y;

  // Recenter on plate (XZ)
  viewMesh.updateMatrixWorld(true);
  const box2 = new THREE.Box3().setFromObject(viewMesh);
  const center = box2.getCenter(new THREE.Vector3());
  viewMesh.position.x -= center.x;
  viewMesh.position.z -= center.z;

  viewMesh.updateMatrixWorld(true);

  // Load arrow
  const loadAxis = getLoadAxisFromUI(lastAnalysisData);
  const loadWorld = loadAxis.clone().applyQuaternion(currentViewQuat).normalize();
  updateLoadArrow(loadWorld, viewMesh);
}

function rotateView(axis, radians) {
  if (!viewMesh) return;

  const q = new THREE.Quaternion();
  const v = new THREE.Vector3(axis[0], axis[1], axis[2]);

  q.setFromAxisAngle(v, radians);
  viewMesh.quaternion.premultiply(q);
  currentViewQuat.copy(viewMesh.quaternion).normalize();
  hasManualViewRotation = true;
  reflowViewMesh();
}

function getCurrentViewExportQuaternion() {
  if (!viewMesh) return null;
  return qYupToZup.clone().multiply(viewMesh.quaternion.clone()).normalize();
}

function ensureManualRotateUI() {
  if (document.getElementById("rotXPlus")) return;

  const container = document.createElement("div");
  container.className = "manual-rotate";
  container.innerHTML = `
    <button id="rotXPlus" type="button">X+</button>
    <button id="rotXMinus" type="button">X-</button>
    <button id="rotYPlus" type="button">Y+</button>
    <button id="rotYMinus" type="button">Y-</button>
    <button id="rotZPlus" type="button">Z+</button>
    <button id="rotZMinus" type="button">Z-</button>
    <button id="rotReset" type="button">Reset</button>
  `;

  const anchor =
    document.getElementById("toggleViewBtn") ||
    document.getElementById("exportBtn") ||
    document.getElementById("fileInput");

  const parent = anchor?.parentElement || document.body;
  parent.appendChild(container);
}


function getRotateStepRadians() {
  const sel = document.getElementById("rotateStep");
  const deg = Number(sel?.value || 15);
  return deg * Math.PI / 180;
}






function exportAsShownSTL() {
  if (!currentMesh) return;

  const exporter = new STLExporter();

  // 1️⃣ Force all transforms to be current
  currentMesh.updateMatrixWorld(true);

  // 2️⃣ Clone geometry (never touch live geometry)
  const geom = currentMesh.geometry.clone();

  // 3️⃣ Bake EXACT view transform into vertices
  geom.applyMatrix4(currentMesh.matrixWorld);

  // 4️⃣ Drop to bed (Z = 0) and center XY
  geom.computeBoundingBox();
  const bb = geom.boundingBox;

  const tx = -(bb.min.x + bb.max.x) / 2;
  const ty = -(bb.min.y + bb.max.y) / 2;
  const tz = -bb.min.z;

  geom.applyMatrix4(
    new THREE.Matrix4().makeTranslation(tx, ty, tz)
  );

  geom.computeVertexNormals();

  // 5️⃣ Wrap in temp mesh for exporter
  const tempMesh = new THREE.Mesh(geom);

  // 6️⃣ Export
  const data = exporter.parse(tempMesh, { binary: true });
  const buffer = data instanceof DataView ? data.buffer : data;

  const blob = new Blob([buffer], { type: "application/octet-stream" });
  const url = URL.createObjectURL(blob);

  const a = document.createElement("a");
  a.href = url;
  a.download = "oriented_export_as_shown.stl";
  document.body.appendChild(a);
  a.click();
  a.remove();

  URL.revokeObjectURL(url);
}




function exportBakedQuaternion(quat, label) {
  if (!quat || !lastAnalysisData) return;

  const exporter = new STLExporter();

  const geom = lastAnalysisData.originalGeometry.clone();

  geom.applyQuaternion(quat);

  geom.computeBoundingBox();
  const bb = geom.boundingBox;

  const tx = -(bb.min.x + bb.max.x) / 2;
  const ty = -(bb.min.y + bb.max.y) / 2;
  const tz = -bb.min.z;

  geom.applyMatrix4(
    new THREE.Matrix4().makeTranslation(tx, ty, tz)
  );

  geom.computeVertexNormals();

  const mesh = new THREE.Mesh(geom);

  const data = exporter.parse(mesh, { binary: true });
  const buffer = data instanceof DataView ? data.buffer : data;

  const blob = new Blob([buffer], { type: "application/octet-stream" });
  const url = URL.createObjectURL(blob);

  const a = document.createElement("a");
  a.href = url;
  a.download = `oriented_export_${label}.stl`;
  document.body.appendChild(a);
  a.click();
  a.remove();

  URL.revokeObjectURL(url);
}





function exportSTL() {
  const mode = getExportMode();

  if (!currentMesh) return;

  if (mode === "view") {
    const exportQuat = getCurrentViewExportQuaternion();
    if (exportQuat) exportBakedQuaternion(exportQuat, "view");
    return;
  }

  if (mode === "strength") {
    exportBakedQuaternion(bestQuatStrength, "strength");
  } 
  else if (mode === "supports") {
    exportBakedQuaternion(bestQuatSupports, "supports");
  }
}



document.getElementById("toggleViewBtn")?.addEventListener("click", () => {
  if (!viewMesh) return;

  viewMode = viewMode === "strength" ? "supports" : "strength";

  const btn = document.getElementById("toggleViewBtn");
  if (btn) btn.textContent = `View: ${viewMode === "strength" ? "Strength" : "Supports"}`;

  showCandidate(0);
});

ensureManualRotateUI();
updateCandidateUI();

document.getElementById("rotXPlus").onclick = () => rotateView([1, 0, 0],  getRotateStepRadians());
document.getElementById("rotXMinus").onclick = () => rotateView([1, 0, 0], -getRotateStepRadians());

document.getElementById("rotYPlus").onclick = () => rotateView([0, 1, 0],  getRotateStepRadians());
document.getElementById("rotYMinus").onclick = () => rotateView([0, 1, 0], -getRotateStepRadians());

document.getElementById("rotZPlus").onclick = () => rotateView([0, 0, 1],  getRotateStepRadians());
document.getElementById("rotZMinus").onclick = () => rotateView([0, 0, 1], -getRotateStepRadians());

document.getElementById("rotReset")?.addEventListener("click", () => {
  const quat = getCurrentAnalysisQuaternion();
  if (quat) applyViewQuaternion(quat);
});

document.getElementById("candidatePrevBtn")?.addEventListener("click", () => {
  showCandidate(currentCandidateIndex - 1);
});

document.getElementById("candidateNextBtn")?.addEventListener("click", () => {
  showCandidate(currentCandidateIndex + 1);
});


// ---------------------------------------- EXPORT BUTTON ------------------------------
document.getElementById("exportBtn").onclick = exportSTL;




// ---------------------------------------- CLEAR BUILDPLATE BUTTON --------------------
document.getElementById("clearBuildplateBtn")?.addEventListener(
  "click",
  clearBuildplate
);




document.getElementById("helpBtn")?.addEventListener("click", () => {
  console.log("Help button clicked");

  const modal = document.getElementById("helpModal");
  if (!modal) {
    console.warn("helpModal not found");
    return;
  }

  modal.classList.add("show");
  modal.removeAttribute("hidden");
  modal.setAttribute("aria-hidden", "false");
});



document.getElementById("helpOk")?.addEventListener("click", closeHelp);
document.getElementById("helpClose")?.addEventListener("click", closeHelp);




//------------------------------ LOAD DIRECTION ARROW ------------------------------

function updateLoadArrow(directionWorld, mesh) {
  if (!mesh) return;

  // Normalize direction
  const dir = directionWorld.clone().normalize();

  // Pick an arrow length based on model size
  const box = new THREE.Box3().setFromObject(mesh);
  const size = box.getSize(new THREE.Vector3()).length();
  const length = Math.max(10, size * 0.35);

  // Start arrow at mesh center
  const center = box.getCenter(new THREE.Vector3());

  if (!loadArrow) {
    // Create a new ArrowHelper
    loadArrow = new THREE.ArrowHelper(dir, center, length, 0xff4d6d);
    scene.add(loadArrow);
  } else {
    // Update existing ArrowHelper
    loadArrow.position.copy(center);
    loadArrow.setDirection(dir);
    loadArrow.setLength(length, length * 0.25, length * 0.12);
  }
}




function getLoadAxisFromUI(analysisData) {
  const sel = normalizeAxisSelection(getLoadAxisSelectorValue());

  switch (sel) {
    case "+x": return new THREE.Vector3(1, 0, 0);
    case "-x": return new THREE.Vector3(-1, 0, 0);
    case "+y": return new THREE.Vector3(0, 1, 0);
    case "-y": return new THREE.Vector3(0, -1, 0);
    case "+z": return new THREE.Vector3(0, 0, 1);
    case "-z": return new THREE.Vector3(0, 0, -1);
    case "principal":
    default:
      return analysisData.principalAxis.clone();
  }
}






// ---------- BASIC SCENE SETUP ----------

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b1020);

const camera = new THREE.PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  0.1,
  1000
);
camera.position.set(0, 60, 120);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

// ---------- CONTROLS ----------
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

// ---------- LIGHTING ----------
scene.add(new THREE.AmbientLight(0xffffff, 0.35));

const keyLight = new THREE.DirectionalLight(0xffffff, 0.8);
keyLight.position.set(100, 100, 100);
scene.add(keyLight);

// ---------- GRID + AXES ----------
const grid = new THREE.GridHelper(200, 20, 0x2a3fff, 0x1b255a);
scene.add(grid);

const axes = new THREE.AxesHelper(50);
scene.add(axes);

// ---------- STL LOADING ----------
const loader = new STLLoader();
let currentMesh = null;



document.getElementById("fileInput").addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = () => {
    loadSTL(reader.result);
  };
  reader.readAsArrayBuffer(file);
});


// ---------------------- LOAD AND ANALYZE STL ----------------------
function loadSTL(buffer) {
  lastSTLBuffer = buffer;

  // Parse STL
  const parsed = loader.parse(buffer);
  parsed.computeVertexNormals();
  parsed.center();

  // Store pristine base geometry
  baseGeometry = parsed.clone();

  // Analysis data
  const analysisData = preprocessGeometry(baseGeometry);
  lastAnalysisData = analysisData;

  // Inspector must run AFTER analysisData exists
  const inspector = computeInspectorReport(analysisData);
  window.lastInspectorReport = inspector;
  console.log("Inspector:", inspector);
  renderInspectorHUD(inspector);

  console.log("Triangle count:", analysisData.triangles.length);
  console.log("Total surface area:", analysisData.totalArea.toFixed(2));

  // Candidate orientations
  const candidates = buildCandidates(42);

  // Load axis from UI
  const loadAxis = getLoadAxisFromUI(analysisData);
  console.log("Load axis:", loadAxis.toArray());

  // Strength pass
  const strengthScores = candidates.map((c, idx) => {
    const s = scoreStrength(analysisData, c.matrix, c.quaternion, loadAxis);
    return { idx, score: s, quat: c.quaternion };
  });
  strengthScores.sort((a, b) => a.score - b.score);
  bestQuatStrength = strengthScores[0].quat.clone();

  console.log("Best strength scores (lower is better):");
  console.table(strengthScores.slice(0, 5));

  // Supports pass
  const supportsScores = candidates.map((c, idx) => {
    const s = scoreOverhang(analysisData.triangles, c.matrix, c.quaternion);
    return { idx, score: s, quat: c.quaternion };
  });
  supportsScores.sort((a, b) => a.score - b.score);
  bestQuatSupports = supportsScores[0].quat.clone();
  rebuildRankedCandidates(strengthScores, supportsScores);

  console.log("Best supports/overhang scores (lower is better):");
  console.table(supportsScores.slice(0, 5));
  console.log("bestQuatStrength", bestQuatStrength.toArray());
  console.log("bestQuatSupports", bestQuatSupports.toArray());
  console.log("dot similarity", Math.abs(bestQuatStrength.dot(bestQuatSupports)));


  // Build or replace the view mesh (from pristine base)
  if (viewMesh) {
    scene.remove(viewMesh);
    viewMesh.geometry.dispose();
    viewMesh.material.dispose();
    viewMesh = null;
  }

  const material = new THREE.MeshStandardMaterial({
    color: 0x8ef0ff,
    metalness: 0.1,
    roughness: 0.35,
  });

  viewMesh = new THREE.Mesh(baseGeometry.clone(), material);
  scene.add(viewMesh);

  // Export uses currentMesh, so keep it aligned with what we're seeing
  currentMesh = viewMesh;

  console.log("Model loaded and added to scene.", {
    triangles: analysisData?.triangles?.length,
    watertight: window.lastInspectorReport?.edgeStats?.watertight,
  });

  // ---- Default view mode on load ----
  // Always start in Strength view unless you explicitly change it elsewhere.
  viewMode = "strength";
const btn = document.getElementById("toggleViewBtn");
if (btn) btn.textContent = "View: Strength";



  // Apply view based on viewMode
  showCandidate(0);

  // Auto-frame camera
  const box = new THREE.Box3().setFromObject(viewMesh);
  const size = box.getSize(new THREE.Vector3()).length();
  const center = box.getCenter(new THREE.Vector3());

  controls.target.copy(center);
  camera.position.set(center.x, center.y + size * 0.6, center.z + size);
  camera.lookAt(center);
}


function clearBuildplate() {
  // Remove visible mesh
  if (viewMesh) {
    scene.remove(viewMesh);
    viewMesh.geometry.dispose();
    viewMesh.material.dispose();
    viewMesh = null;
  }

  // Clear analysis + state
  lastSTLBuffer = null;
  lastAnalysisData = null;
  baseGeometry = null;
  bestQuatStrength = null;
  bestQuatSupports = null;
  hasManualViewRotation = false;
  rankedCandidatesStrength = [];
  rankedCandidatesSupports = [];
  currentCandidateIndex = 0;
  updateCandidateUI();

  // Clear inspector HUD
  const hud = document.getElementById("inspectorHud");
  if (hud) hud.hidden = true;

  // Clear load arrow
  if (loadArrow) {
    scene.remove(loadArrow);
    loadArrow = null;
  }

  // Reset camera to neutral
  controls.target.set(0, 0, 0);
  camera.position.set(0, 60, 120);
  camera.lookAt(0, 0, 0);

  console.log("Buildplate cleared");
}



// ---------- RENDER LOOP ----------
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();

// ---------- RESIZE ----------
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

document.getElementById("reAnalyzeBtn")?.addEventListener("click", () => {
  if (lastSTLBuffer) loadSTL(lastSTLBuffer);
});

const fileNameEl = document.getElementById("fileName");
document.getElementById("fileInput")?.addEventListener("change", (e) => {
  const f = e.target.files?.[0];
  if (fileNameEl) fileNameEl.textContent = f ? f.name : "No file";
});
