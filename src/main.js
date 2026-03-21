import "./style.css";
import * as THREE from "three";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { preprocessGeometry } from "./analyze/preprocess.js";
import { buildCandidates } from "./analyze/candidates.js";
import { scoreOverhang } from "./analyze/scoreOverhang.js";
import { scoreStrength } from "./analyze/scoreStrength.js";
import { scoreStructuralRisk } from "./analyze/scoreStructuralRisk.js";
import { STLExporter } from "three/examples/jsm/exporters/STLExporter.js";
import { computeInspectorReport } from "./analyze/inspector.js";
import { analyzeOrientationRecommendations } from "./analyze/orientationRecommendations.js";
import { buildOverhangVertexColors, buildSupportShadowSegments } from "./analyze/overhangHeatmap.js";



let lastSTLBuffer = null;
let lastAnalysisData = null;

let baseGeometry = null;   // pristine, never rotated
let viewMesh = null;       // what the user sees

let bestQuatStrength = null;
let bestQuatSupports = null;

let loadArrow = null;      // your arrow helper
let currentViewQuat = new THREE.Quaternion(); // tracks what is displayed

let lastFocusEl = null;

let viewMode = "strength"; // "strength" | "supports" | "overhang"
let rankedCandidatesStrength = [];
let rankedCandidatesSupports = [];
let currentCandidateIndex = 0;
let orientationRecommendations = [];
let overhangColorBuffer = null;
let overhangHeatmapMode = "angle";
let supportShadowsEnabled = false;
let supportShadowOverlay = null;

const VIEW_MODES = ["strength", "supports", "overhang"];
const DEFAULT_VIEW_COLOR = 0x8ef0ff;

function getCandidateRankingMode(mode = viewMode) {
  return mode === "strength" ? "strength" : "supports";
}

function isSupportVisualizationMode(mode = viewMode) {
  return mode === "supports" || mode === "overhang";
}

function usesOverhangHeatmap(mode = viewMode) {
  return mode === "overhang";
}


function openHelp() {
  const helpModal = document.getElementById("helpModal");
  const helpClose = document.getElementById("helpClose");
  if (!helpModal) return;

  lastFocusEl = document.activeElement;

  helpModal.removeAttribute("hidden");
  helpModal.classList.add("open");    // pick one system
  helpModal.classList.remove("show"); // optional cleanup
  helpModal.setAttribute("aria-hidden", "false");

  helpClose?.focus(); // or helpOk?.focus()
}


function closeHelp() {
  const helpModal = document.getElementById("helpModal");
  // Pick a safe focus target outside the modal
  const fallback = document.getElementById("helpBtn");
  const dontShowAgain = document.getElementById("dontShowAgain");
  if (!helpModal) return;

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
    localStorage.setItem("pc_help_seen_v1", "1");
  }
}

function wireHelpModal() {
  document.getElementById("helpBtn")?.addEventListener("click", openHelp);
  document.getElementById("helpOk")?.addEventListener("click", closeHelp);
  document.getElementById("helpClose")?.addEventListener("click", closeHelp);
  document.querySelector("#helpModal .modal-backdrop")?.addEventListener("click", closeHelp);

  window.addEventListener("keydown", (event) => {
    const helpModal = document.getElementById("helpModal");
    if (event.key === "Escape" && helpModal?.classList.contains("open")) {
      closeHelp();
    }
  });
}

function enforceCtrlWheelZoom(domElement, orbitControls) {
  orbitControls.enableZoom = false;

  domElement.addEventListener(
    "wheel",
    (event) => {
      orbitControls.enableZoom = event.ctrlKey;

      if (!event.ctrlKey) return;

      event.preventDefault();
    },
    { passive: false, capture: true }
  );
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
  return getCandidateRankingMode() === "strength" ? bestQuatStrength : bestQuatSupports;
}

function getLoadAxisSelectorValue() {
  const el =
    document.getElementById("loadAxisSelect") ||
    document.getElementById("loadDir") ||
    document.getElementById("loadAxis") ||
    document.querySelector("[name='loadDir']") ||
    document.querySelector("[name='loadAxis']");

  return el?.value ?? "principal";
}

function getLoadTypeSelectorValue() {
  const el =
    document.getElementById("loadTypeSelect") ||
    document.getElementById("loadType") ||
    document.querySelector("[name='loadType']");

  return el?.value ?? "compression";
}

function normalizeLoadTypeSelection(value) {
  switch ((value ?? "").trim().toLowerCase()) {
    case "bending":
      return "bending";
    case "shear":
      return "shear";
    case "tension":
      return "tension";
    case "compression":
    default:
      return "compression";
  }
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
  return getCandidateRankingMode() === "strength" ? rankedCandidatesStrength : rankedCandidatesSupports;
}

function getViewModeLabel(mode = viewMode) {
  switch (mode) {
    case "supports":
      return "Supports";
    case "overhang":
      return "Overhang";
    case "strength":
    default:
      return "Strength";
  }
}

function updateViewModeButton() {
  const btn = document.getElementById("toggleViewBtn");
  if (btn) btn.textContent = `View: ${getViewModeLabel()}`;
}

function updateHeatmapMetricUI() {
  const group = document.getElementById("overhangMetricGroup");
  const select = document.getElementById("overhangMetric");
  const shadowToggle = document.getElementById("supportShadowsToggle");

  if (group) group.hidden = !isSupportVisualizationMode();
  if (select && select.value !== overhangHeatmapMode) {
    select.value = overhangHeatmapMode;
  }
  if (shadowToggle) shadowToggle.checked = supportShadowsEnabled;
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
      ? `STR ${cand.strengthScore?.toFixed(1)} | SUPV ${cand.supportsScore?.toFixed(0)}`
      : "STR - | SUPV -";
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

function rebuildRankedCandidates(candidateAnalyses, strengthScores, supportsScores) {
  const strengthByIdx = new Map(strengthScores.map((item) => [item.idx, item.score]));
  const supportsByIdx = new Map(supportsScores.map((item) => [item.idx, item.score]));
  const structuralRiskByIdx = new Map(
    candidateAnalyses.map((item) => [item.idx, item.structuralRisk])
  );

  const buildRankedList = (sorted) =>
    sorted.slice(0, 5).map((item, rank) => ({
      rank: rank + 1,
      quaternion: item.quat?.clone?.() || null,
      strengthScore: strengthByIdx.get(item.idx),
      supportsScore: supportsByIdx.get(item.idx),
      structuralRisk: structuralRiskByIdx.get(item.idx) ?? null,
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

function formatPercent(value, invert = false) {
  const pct = Math.round((invert ? 1 - value : value) * 100);
  return `${Math.max(0, Math.min(100, pct))}%`;
}

function resetOrientationPanel(message = "No recommendations yet.") {
  orientationRecommendations = [];

  const panel = document.getElementById("orientationPanel");
  const results = document.getElementById("orientationResults");
  const meta = document.getElementById("orientationMeta");

  if (panel) panel.hidden = false;
  if (meta) meta.textContent = message;
  if (results) {
    results.innerHTML = `<div class="orientationEmpty">${message}</div>`;
  }
}

function renderOrientationRecommendations(result) {
  orientationRecommendations = result.recommendations;

  const panel = document.getElementById("orientationPanel");
  const results = document.getElementById("orientationResults");
  const meta = document.getElementById("orientationMeta");

  if (!panel || !results || !meta) return;

  panel.hidden = false;
  meta.textContent = `${result.candidateCount} candidates sampled at ${result.stepDegrees}° steps`;

  results.innerHTML = "";

  for (const recommendation of result.recommendations) {
    const metricLine = recommendation.structuralRisk
      ? `
        <span>Risk ${recommendation.metrics.structuralRiskPercent}%</span>
        <span>${recommendation.structuralRisk.label}</span>
        <span>Type ${recommendation.structuralRisk.loadType}</span>
      `
      : `
        <span>SUPV ${Math.round(recommendation.metrics.supportVolume ?? 0)}</span>
        <span>Stability ${formatPercent(recommendation.metrics.bedStability)}</span>
        <span>Height ${recommendation.metrics.printHeight.toFixed(1)}</span>
      `;

    const card = document.createElement("div");
    card.className = "orientationCard";
    card.innerHTML = `
      <div class="orientationCardHead">
        <div class="orientationLabel">${recommendation.label}</div>
        <div class="orientationAngles">X ${recommendation.xDeg}° / Y ${recommendation.yDeg}°</div>
      </div>
      <div class="orientationMetrics">
        ${metricLine}
      </div>
      ${recommendation.explanation ? `<div class="orientationNote">${recommendation.explanation}</div>` : ""}
      <button class="btn orientationApply" type="button">Apply</button>
    `;

    card.querySelector(".orientationApply")?.addEventListener("click", () => {
      applyViewQuaternion(recommendation.quaternion);
    });

    results.appendChild(card);
  }
}

function refreshViewMeshColors() {
  if (!viewMesh || !lastAnalysisData) return;

  const material = viewMesh.material;
  if (!(material instanceof THREE.MeshStandardMaterial)) return;

  if (!usesOverhangHeatmap()) {
    material.vertexColors = false;
    material.color.setHex(DEFAULT_VIEW_COLOR);
    material.needsUpdate = true;
    return;
  }

  const zUpQuat = getCurrentViewExportQuaternion();
  if (!zUpQuat) return;

  const rotMatrix = new THREE.Matrix4().makeRotationFromQuaternion(zUpQuat);

  overhangColorBuffer = buildOverhangVertexColors(
    lastAnalysisData.triangles,
    rotMatrix,
    {
      mode: overhangHeatmapMode,
      bbox: lastAnalysisData.bbox,
    },
    overhangColorBuffer
  );

  let colorAttr = viewMesh.geometry.getAttribute("color");
  if (!colorAttr || colorAttr.array.length !== overhangColorBuffer.length) {
    colorAttr = new THREE.BufferAttribute(overhangColorBuffer, 3);
    viewMesh.geometry.setAttribute("color", colorAttr);
  } else {
    colorAttr.array.set(overhangColorBuffer);
    colorAttr.needsUpdate = true;
  }

  material.vertexColors = true;
  material.color.setHex(0xffffff);
  material.needsUpdate = true;
}

function refreshSupportShadowOverlay() {
  if (!viewMesh || !lastAnalysisData || !isSupportVisualizationMode() || !supportShadowsEnabled) {
    if (supportShadowOverlay) {
      supportShadowOverlay.visible = false;
    }
    return;
  }

  if (supportShadowOverlay) {
    supportShadowOverlay.visible = true;
  }

  const zUpQuat = getCurrentViewExportQuaternion();
  if (!zUpQuat) return;

  const rotMatrix = new THREE.Matrix4().makeRotationFromQuaternion(zUpQuat);

  const { positions, colors } = buildSupportShadowSegments(lastAnalysisData.triangles, rotMatrix, {
    mode: overhangHeatmapMode,
    bbox: lastAnalysisData.bbox,
    severityThreshold: 0.22,
    maxSegments: 600,
  });

  if (!supportShadowOverlay) {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

    const material = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.34,
      depthTest: false,
      depthWrite: false,
    });

    supportShadowOverlay = new THREE.LineSegments(geometry, material);
    supportShadowOverlay.quaternion.copy(qZupToYup);
    supportShadowOverlay.renderOrder = 4;
    scene.add(supportShadowOverlay);
    return;
  }

  supportShadowOverlay.geometry.dispose();
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  supportShadowOverlay.geometry = geometry;
}


function estimateSupportVolumeForQuaternion(analysisData, quat, options = {}) {
  if (!analysisData || !quat) return 0;

  const rotMatrix = new THREE.Matrix4().makeRotationFromQuaternion(quat);

  const { totalVolume } = buildSupportShadowSegments(
    analysisData.triangles,
    rotMatrix,
    {
      mode: options.mode ?? "angle-distance",
      bbox: analysisData.bbox,
      severityThreshold: options.severityThreshold ?? 0.22,
      maxSegments: options.maxSegments ?? 600,
    }
  );

  return totalVolume ?? 0;
}



function runOrientationAnalysis() {
  if (!lastAnalysisData) {
    resetOrientationPanel("Load an STL to analyze orientation.");
    return;
  }

  const result = analyzeOrientationRecommendations(lastAnalysisData, {
    stepDegrees: 45,
    loadAxis: getLoadAxisFromUI(lastAnalysisData),
    loadType: normalizeLoadTypeSelection(getLoadTypeSelectorValue()),
  });
  renderOrientationRecommendations(result);
}


const toggleViewBtn = document.getElementById("toggleViewBtn");





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
  refreshViewMeshColors();
  refreshSupportShadowOverlay();
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
  if (!currentMesh) return;

  const exportQuat = getCurrentViewExportQuaternion();
  if (!exportQuat) return;

  exportBakedQuaternion(exportQuat, "view");
}



document.getElementById("toggleViewBtn")?.addEventListener("click", () => {
  const previousCandidateMode = getCandidateRankingMode();
  const modeIndex = VIEW_MODES.indexOf(viewMode);
  viewMode = VIEW_MODES[(modeIndex + 1) % VIEW_MODES.length];
  const nextCandidateMode = getCandidateRankingMode();
  updateViewModeButton();
  updateHeatmapMetricUI();

  if (!viewMesh) return;

  if (previousCandidateMode !== nextCandidateMode) {
    showCandidate(0);
    return;
  }

  refreshViewMeshColors();
  refreshSupportShadowOverlay();
  updateCandidateUI();
});

ensureManualRotateUI();
updateCandidateUI();
updateViewModeButton();
updateHeatmapMetricUI();
resetOrientationPanel("Load an STL to analyze orientation.");

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

document.getElementById("analyzeOrientationBtn")?.addEventListener("click", runOrientationAnalysis);
document.getElementById("overhangMetric")?.addEventListener("change", (event) => {
  overhangHeatmapMode = event.target.value;
  if (!isSupportVisualizationMode()) return;
  refreshViewMeshColors();
  refreshSupportShadowOverlay();
});
document.getElementById("supportShadowsToggle")?.addEventListener("change", (event) => {
  supportShadowsEnabled = event.target.checked;
  refreshSupportShadowOverlay();
});
document.getElementById("loadAxisSelect")?.addEventListener("change", refreshAnalysisForLoadInputs);
document.getElementById("loadTypeSelect")?.addEventListener("change", refreshAnalysisForLoadInputs);


// ---------------------------------------- EXPORT BUTTON ------------------------------
document.getElementById("exportBtn").onclick = exportSTL;




// ---------------------------------------- CLEAR BUILDPLATE BUTTON --------------------
document.getElementById("clearBuildplateBtn")?.addEventListener(
  "click",
  clearBuildplate
);




wireHelpModal();




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
enforceCtrlWheelZoom(renderer.domElement, controls);

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


function quickOrientationRiskCheck() {

  if (!lastAnalysisData) return;

  const rotMatrix = new THREE.Matrix4().identity();

  const supportScore = scoreOverhang(
    lastAnalysisData.triangles,
    rotMatrix
  );

  // crude heuristic threshold
  if (supportScore > 250000) {
    showOrientationWarning();
  }

}


function showOrientationWarning() {

  if (document.getElementById("orientationWarn")) return;

  const warn = document.createElement("div");
  warn.id = "orientationWarn";
  warn.className = "orientationWarn";

  warn.innerHTML = `
    ⚠ Current orientation may require heavy supports.
    <button id="warnAnalyze">Analyze Orientation</button>
  `;

  document.body.appendChild(warn);

  document.getElementById("warnAnalyze")?.addEventListener("click", () => {
    document.getElementById("analyzeOrientationBtn")?.click();
    warn.remove();
  });

}


// ---------------------- LOAD AND ANALYZE STL ----------------------
function loadSTL(buffer) {
  lastSTLBuffer = buffer;
  overhangColorBuffer = null;

  // Parse STL
  const parsed = loader.parse(buffer);
  parsed.computeVertexNormals();
  parsed.center();

  // Store pristine base geometry
  baseGeometry = parsed.clone();

  // Analysis data
  const analysisData = preprocessGeometry(baseGeometry);
  lastAnalysisData = analysisData;

  // Quick orientation risk check
  quickOrientationRiskCheck();

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
  const loadType = normalizeLoadTypeSelection(getLoadTypeSelectorValue());
  console.log("Load axis:", loadAxis.toArray());
  console.log("Load type:", loadType);

  const candidateAnalyses = candidates.map((candidate, idx) => {
    const strengthScore = scoreStrength(
      analysisData,
      candidate.matrix,
      candidate.quaternion,
      loadAxis
    );

    const overhangScore = scoreOverhang(
      analysisData.triangles,
      candidate.matrix,
      candidate.quaternion
    );

    const supportVolume = estimateSupportVolumeForQuaternion(
      analysisData,
      candidate.quaternion,
      {
        mode: overhangHeatmapMode,
        severityThreshold: 0.22,
        maxSegments: 600,
      }
    );

    const structuralRisk = scoreStructuralRisk(
      analysisData,
      candidate.matrix,
      candidate.quaternion,
      loadAxis,
      loadType
    );

    return {
      idx,
      quat: candidate.quaternion,
      strengthScore,
      overhangScore,
      supportVolume,
      supportsScore: supportVolume,
      structuralRisk,
    };
  });

  analysisData.candidateAnalyses = candidateAnalyses.map((item) => ({
    idx: item.idx,
    quaternion: item.quat.clone(),
    strengthScore: item.strengthScore,
    overhangScore: item.overhangScore,
    supportVolume: item.supportVolume,
    supportsScore: item.supportsScore,
    structuralRisk: {
      normalizedScore: item.structuralRisk.normalizedScore,
      label: item.structuralRisk.label,
      loadType: item.structuralRisk.loadType,
      breakdown: { ...item.structuralRisk.breakdown },
    },
  }));

  // Strength pass
  const strengthScores = candidateAnalyses.map((item) => ({
    idx: item.idx,
    score: item.strengthScore,
    quat: item.quat,
  }));
  strengthScores.sort((a, b) => a.score - b.score);
  bestQuatStrength = strengthScores[0].quat.clone();

  console.log("Best strength scores (lower is better):");
  console.table(strengthScores.slice(0, 5));

  // Supports pass
  const supportsScores = candidateAnalyses.map((item) => ({
    idx: item.idx,
    score: item.supportsScore,
    quat: item.quat,
  }));
  supportsScores.sort((a, b) => a.score - b.score);
  bestQuatSupports = supportsScores[0].quat.clone();
  rebuildRankedCandidates(candidateAnalyses, strengthScores, supportsScores);

  console.log("Best support-volume scores (lower is better):");
  console.table(supportsScores.slice(0, 5));
  console.log("Lowest structural-risk scores (lower is better):");
  console.table(
    [...candidateAnalyses]
      .sort((a, b) => a.structuralRisk.normalizedScore - b.structuralRisk.normalizedScore)
      .slice(0, 5)
      .map((item) => ({
        idx: item.idx,
        score: item.structuralRisk.normalizedScore.toFixed(3),
        label: item.structuralRisk.label,
      }))
  );
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

  if (supportShadowOverlay) {
    scene.remove(supportShadowOverlay);
    supportShadowOverlay.geometry.dispose();
    supportShadowOverlay.material.dispose();
    supportShadowOverlay = null;
  }

  const material = new THREE.MeshStandardMaterial({
    color: DEFAULT_VIEW_COLOR,
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
  updateViewModeButton();
  updateHeatmapMetricUI();



  // Apply view based on viewMode
  showCandidate(0);
  resetOrientationPanel("Click Analyze Orientation to generate recommendations.");

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
  overhangColorBuffer = null;
  rankedCandidatesStrength = [];
  rankedCandidatesSupports = [];
  currentCandidateIndex = 0;
  viewMode = "strength";
  updateCandidateUI();
  updateViewModeButton();
  updateHeatmapMetricUI();
  resetOrientationPanel("Load an STL to analyze orientation.");

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


// AUTO SHOW HELP ON FIRST VISIT
if (!localStorage.getItem("pc_help_seen_v1")) {
  openHelp();
}

function refreshAnalysisForLoadInputs() {
  if (lastSTLBuffer) loadSTL(lastSTLBuffer);
}
