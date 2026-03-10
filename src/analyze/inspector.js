// src/analyze/inspector.js
// STL Inspector Pro v1 integrated module
// Consumes: analysisData from preprocessGeometry()
// Produces: inspector report with edge health, shells, volume, degenerates, unit guess, score

import * as THREE from "three";

// Quantize for stable hashing
function hashV3(v, eps = 1e-5) {
  const qx = Math.round(v.x / eps);
  const qy = Math.round(v.y / eps);
  const qz = Math.round(v.z / eps);
  return `${qx},${qy},${qz}`;
}

// This tries to support common triangle formats.
// Adjust this if your preprocessGeometry uses a different shape.
function triVerts(tri) {
  // Format A: { a:Vector3, b:Vector3, c:Vector3 }
  if (tri?.a && tri?.b && tri?.c) return [tri.a, tri.b, tri.c];

  // Format B: [Vector3, Vector3, Vector3]
  if (Array.isArray(tri) && tri.length >= 3) return [tri[0], tri[1], tri[2]];

  // Format C: { v0, v1, v2 }
  if (tri?.v0 && tri?.v1 && tri?.v2) return [tri.v0, tri.v1, tri.v2];

  // Format D: flat numeric arrays
  // Example: { ax, ay, az, bx, by, bz, cx, cy, cz }
  if (Number.isFinite(tri?.ax)) {
    return [
      new THREE.Vector3(tri.ax, tri.ay, tri.az),
      new THREE.Vector3(tri.bx, tri.by, tri.bz),
      new THREE.Vector3(tri.cx, tri.cy, tri.cz),
    ];
  }

  throw new Error("Inspector: unknown triangle format. Update triVerts() to match preprocessGeometry output.");
}

function triArea(a, b, c) {
  const ab = new THREE.Vector3().subVectors(b, a);
  const ac = new THREE.Vector3().subVectors(c, a);
  return 0.5 * new THREE.Vector3().crossVectors(ab, ac).length();
}

// Signed volume contribution of tetrahedron (0, a, b, c)
function triSignedVolume(a, b, c) {
  return a.dot(new THREE.Vector3().crossVectors(b, c)) / 6.0;
}

// Union Find for shells
class DSU {
  constructor(n) {
    this.p = Array.from({ length: n }, (_, i) => i);
    this.r = new Array(n).fill(0);
  }
  find(x) {
    while (this.p[x] !== x) {
      this.p[x] = this.p[this.p[x]];
      x = this.p[x];
    }
    return x;
  }
  union(a, b) {
    a = this.find(a);
    b = this.find(b);
    if (a === b) return;
    if (this.r[a] < this.r[b]) [a, b] = [b, a];
    this.p[b] = a;
    if (this.r[a] === this.r[b]) this.r[a]++;
  }
}

export function computeInspectorReport(analysisData) {
  const triangles = analysisData?.triangles || [];
  const triCount = triangles.length;

  // BBox and size
  const bbox = analysisData?.bbox || null;
  const size = bbox
    ? {
        x: bbox.max.x - bbox.min.x,
        y: bbox.max.y - bbox.min.y,
        z: bbox.max.z - bbox.min.z,
      }
    : null;

  // Edge health and shells
  const edgeCount = new Map(); // undirected edge key -> count
  const dsu = new DSU(triCount);
  const vertexOwner = new Map(); // vertex hash -> triangle index

  let surfaceArea = 0;
  let signedVolume = 0;
  let degenerateTriangles = 0;

  for (let i = 0; i < triCount; i++) {
    const [a, b, c] = triVerts(triangles[i]);

    const area = triArea(a, b, c);
    surfaceArea += area;
    if (!Number.isFinite(area) || area === 0) degenerateTriangles++;

    signedVolume += triSignedVolume(a, b, c);

    const ha = hashV3(a);
    const hb = hashV3(b);
    const hc = hashV3(c);

    // Edge keys
    const edges = [
      [ha, hb],
      [hb, hc],
      [hc, ha],
    ];

    for (const [u0, v0] of edges) {
      const u = u0 < v0 ? u0 : v0;
      const v = u0 < v0 ? v0 : u0;
      const key = `${u}|${v}`;
      edgeCount.set(key, (edgeCount.get(key) || 0) + 1);
    }

    // Shell unions by shared vertices
    for (const h of [ha, hb, hc]) {
      if (vertexOwner.has(h)) dsu.union(i, vertexOwner.get(h));
      else vertexOwner.set(h, i);
    }
  }

  let boundaryEdges = 0;
  let nonManifoldEdges = 0;
  for (const c of edgeCount.values()) {
    if (c === 1) boundaryEdges++;
    else if (c > 2) nonManifoldEdges++;
  }
  const watertight = boundaryEdges === 0 && nonManifoldEdges === 0;

  // Shell count
  const roots = new Set();
  for (let i = 0; i < triCount; i++) roots.add(dsu.find(i));
  const shells = roots.size;

  // Volume
  const volume = Math.abs(signedVolume);

  // Unit guess heuristic
  const maxDim = size ? Math.max(size.x, size.y, size.z) : NaN;
  let unitGuess = "unknown";
  if (Number.isFinite(maxDim)) {
    if (maxDim > 1 && maxDim < 1500) unitGuess = "mm (likely)";
    else if (maxDim > 0.05 && maxDim < 60) unitGuess = "inches (possible)";
  }

  // Simple score
  let score = 100;
  if (!watertight) {
    if (boundaryEdges > 0) score -= 25;
    if (nonManifoldEdges > 0) score -= Math.min(25, nonManifoldEdges);
  }
  if (degenerateTriangles > 0) score -= Math.min(15, Math.ceil(degenerateTriangles / 1000));
  if (shells > 1) score -= Math.min(15, shells - 1);
  score = Math.max(0, Math.min(100, score));

  return {
    triangles: triCount,
    bbox,
    size,
    surfaceArea,
    volume,
    degenerateTriangles,
    shells,
    edgeStats: { watertight, boundaryEdges, nonManifoldEdges },
    unitGuess,
    score,
  };
}
