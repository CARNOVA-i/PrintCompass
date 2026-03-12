import * as THREE from "three";

const GRADIENT_STOPS = [
  { t: 0.0, color: new THREE.Color(0x39d5ff) },
  { t: 0.25, color: new THREE.Color(0x2f6fff) },
  { t: 0.55, color: new THREE.Color(0xffd54a) },
  { t: 0.78, color: new THREE.Color(0xff8a3d) },
  { t: 1.0, color: new THREE.Color(0xff4d4d) },
];
const EPSILON = 1e-6;
const SHADOW_LOW = new THREE.Color(0xffad66);
const SHADOW_HIGH = new THREE.Color(0xff5a5a);

function clamp01(value) {
  return Math.max(0, Math.min(1, value));
}

export function getRotatedBounds(bbox, zUpQuat) {
  const matrix = new THREE.Matrix4().makeRotationFromQuaternion(zUpQuat);
  const corners = [
    new THREE.Vector3(bbox.min.x, bbox.min.y, bbox.min.z),
    new THREE.Vector3(bbox.min.x, bbox.min.y, bbox.max.z),
    new THREE.Vector3(bbox.min.x, bbox.max.y, bbox.min.z),
    new THREE.Vector3(bbox.min.x, bbox.max.y, bbox.max.z),
    new THREE.Vector3(bbox.max.x, bbox.min.y, bbox.min.z),
    new THREE.Vector3(bbox.max.x, bbox.min.y, bbox.max.z),
    new THREE.Vector3(bbox.max.x, bbox.max.y, bbox.min.z),
    new THREE.Vector3(bbox.max.x, bbox.max.y, bbox.max.z),
  ];

  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  let minZ = Infinity;
  let maxZ = -Infinity;

  for (const corner of corners) {
    corner.applyMatrix4(matrix);
    minX = Math.min(minX, corner.x);
    maxX = Math.max(maxX, corner.x);
    minY = Math.min(minY, corner.y);
    maxY = Math.max(maxY, corner.y);
    minZ = Math.min(minZ, corner.z);
    maxZ = Math.max(maxZ, corner.z);
  }

  return {
    minX,
    maxX,
    minY,
    maxY,
    minZ,
    maxZ,
    width: Math.max(maxX - minX, EPSILON),
    depth: Math.max(maxY - minY, EPSILON),
    height: Math.max(maxZ - minZ, EPSILON),
  };
}

function getGradientColor(severity, target) {
  const clamped = Math.max(0, Math.min(1, severity));

  for (let i = 1; i < GRADIENT_STOPS.length; i++) {
    const prev = GRADIENT_STOPS[i - 1];
    const next = GRADIENT_STOPS[i];

    if (clamped <= next.t) {
      const span = Math.max(1e-6, next.t - prev.t);
      const alpha = (clamped - prev.t) / span;
      return target.copy(prev.color).lerp(next.color, alpha);
    }
  }

  return target.copy(GRADIENT_STOPS[GRADIENT_STOPS.length - 1].color);
}

function computeTriangleSeverity(triangle, zUpQuat, mode, bounds, rotatedNormal, rotatedCentroid) {
  rotatedNormal.copy(triangle.normal).applyQuaternion(zUpQuat).normalize();
  rotatedCentroid
    .copy(triangle.a)
    .add(triangle.b)
    .add(triangle.c)
    .multiplyScalar(1 / 3)
    .applyQuaternion(zUpQuat);

  const downwardSeverity = Math.max(0, -rotatedNormal.z);
  const normalizedDistance = bounds
    ? clamp01((rotatedCentroid.z - bounds.minZ) / bounds.height)
    : 0;

  let severity = downwardSeverity ** 2;

  if (mode === "angle-distance" && bounds) {
    // Preserve a baseline response near the bed while emphasizing tall unsupported regions.
    severity *= 0.35 + (0.65 * Math.pow(normalizedDistance, 0.75));
  }

  return {
    severity,
    downwardSeverity,
    normalizedDistance,
    centroid: rotatedCentroid,
  };
}

export function buildOverhangVertexColors(triangles, zUpQuat, options = {}, targetArray = null) {
  const colors = targetArray && targetArray.length === triangles.length * 9
    ? targetArray
    : new Float32Array(triangles.length * 9);

  const bbox = options.bbox ?? null;
  const rotatedNormal = new THREE.Vector3();
  const rotatedCentroid = new THREE.Vector3();
  const color = new THREE.Color();
  const bounds = bbox ? getRotatedBounds(bbox, zUpQuat) : null;

  for (let i = 0; i < triangles.length; i++) {
    const triangle = triangles[i];
    const { severity } = computeTriangleSeverity(
      triangle,
      zUpQuat,
      options.mode ?? "angle",
      bounds,
      rotatedNormal,
      rotatedCentroid
    );

    getGradientColor(severity, color);

    const offset = i * 9;
    colors[offset] = color.r;
    colors[offset + 1] = color.g;
    colors[offset + 2] = color.b;
    colors[offset + 3] = color.r;
    colors[offset + 4] = color.g;
    colors[offset + 5] = color.b;
    colors[offset + 6] = color.r;
    colors[offset + 7] = color.g;
    colors[offset + 8] = color.b;
  }

  return colors;
}

export function buildSupportShadowSegments(triangles, zUpQuat, options = {}) {
  const bbox = options.bbox ?? null;
  const mode = options.mode ?? "angle";
  const bounds = bbox ? getRotatedBounds(bbox, zUpQuat) : null;
  if (!bounds) return { positions: new Float32Array(0), colors: new Float32Array(0) };

  const rotatedNormal = new THREE.Vector3();
  const rotatedCentroid = new THREE.Vector3();
  const color = new THREE.Color();

  const maxSpan = Math.max(bounds.width, bounds.depth, EPSILON);
  const cellSize = Math.max(maxSpan * 0.03, bounds.height * 0.02, 1e-3);
  const minHeight = Math.max(bounds.height * 0.03, 1e-4);
  const threshold = options.severityThreshold ?? 0.22;
  const maxSegments = options.maxSegments ?? 600;
  const tx = -((bounds.minX + bounds.maxX) * 0.5);
  const ty = -((bounds.minY + bounds.maxY) * 0.5);
  const tz = -bounds.minZ;

  const cells = new Map();

  for (const triangle of triangles) {
    const result = computeTriangleSeverity(
      triangle,
      zUpQuat,
      mode,
      bounds,
      rotatedNormal,
      rotatedCentroid
    );

    const supportHeight = Math.max(0, result.centroid.z - bounds.minZ);
    if (result.severity < threshold || supportHeight < minHeight || result.downwardSeverity <= 0.35) {
      continue;
    }

    const ix = Math.floor((result.centroid.x - bounds.minX) / cellSize);
    const iy = Math.floor((result.centroid.y - bounds.minY) / cellSize);
    const key = `${ix}:${iy}`;
    const prev = cells.get(key);

    if (!prev || supportHeight > prev.height || result.severity > prev.severity) {
      cells.set(key, {
        x: bounds.minX + ((ix + 0.5) * cellSize),
        y: bounds.minY + ((iy + 0.5) * cellSize),
        height: supportHeight,
        severity: result.severity,
      });
    }
  }

  const segments = [...cells.values()]
    .sort((a, b) => b.severity - a.severity || b.height - a.height)
    .slice(0, maxSegments);

  const positions = new Float32Array(segments.length * 6);
  const colors = new Float32Array(segments.length * 6);

  for (let i = 0; i < segments.length; i++) {
    const segment = segments[i];
    const offset = i * 6;
    const x = segment.x + tx;
    const y = segment.y + ty;
    const z0 = bounds.minZ + tz;
    const z1 = bounds.minZ + segment.height + tz;
    const intensity = clamp01(0.25 + (0.75 * segment.severity));

    color.copy(SHADOW_LOW).lerp(SHADOW_HIGH, intensity);

    positions[offset] = x;
    positions[offset + 1] = y;
    positions[offset + 2] = z0;
    positions[offset + 3] = x;
    positions[offset + 4] = y;
    positions[offset + 5] = z1;

    colors[offset] = color.r;
    colors[offset + 1] = color.g;
    colors[offset + 2] = color.b;
    colors[offset + 3] = color.r;
    colors[offset + 4] = color.g;
    colors[offset + 5] = color.b;
  }

  return { positions, colors };
}
