import * as THREE from "three";

/**
 * Power iteration to get dominant eigenvector of a symmetric 3x3 matrix.
 * Matrix is provided as 9 numbers in row-major:
 * [m00,m01,m02, m10,m11,m12, m20,m21,m22]
 */
function dominantEigenvector3(m, iters = 20) {
  let v = new THREE.Vector3(1, 0.5, 0.25).normalize();
  const tmp = new THREE.Vector3();

  for (let i = 0; i < iters; i++) {
    tmp.set(
      m[0] * v.x + m[1] * v.y + m[2] * v.z,
      m[3] * v.x + m[4] * v.y + m[5] * v.z,
      m[6] * v.x + m[7] * v.y + m[8] * v.z
    );
    if (tmp.lengthSq() < 1e-12) break;
    v.copy(tmp.normalize());
  }
  return v;
}

/**
 * Estimate principal axis (dominant direction) using triangle centroids,
 * weighted by triangle area. We sample if the mesh is huge to keep it snappy.
 */
function estimatePrincipalAxis(triangles, maxSamples = 20000) {
  const step = Math.max(1, Math.floor(triangles.length / maxSamples));

  // Weighted mean of centroids
  let sumW = 0;
  const mean = new THREE.Vector3();

  for (let i = 0; i < triangles.length; i += step) {
    const t = triangles[i];
    const c = new THREE.Vector3().addVectors(t.a, t.b).add(t.c).multiplyScalar(1 / 3);
    const w = t.area;

    mean.addScaledVector(c, w);
    sumW += w;
  }
  if (sumW > 0) mean.multiplyScalar(1 / sumW);

  // Weighted covariance
  let c00 = 0, c01 = 0, c02 = 0, c11 = 0, c12 = 0, c22 = 0;

  for (let i = 0; i < triangles.length; i += step) {
    const t = triangles[i];
    const c = new THREE.Vector3().addVectors(t.a, t.b).add(t.c).multiplyScalar(1 / 3);
    const w = t.area;

    const dx = c.x - mean.x;
    const dy = c.y - mean.y;
    const dz = c.z - mean.z;

    c00 += w * dx * dx;
    c01 += w * dx * dy;
    c02 += w * dx * dz;
    c11 += w * dy * dy;
    c12 += w * dy * dz;
    c22 += w * dz * dz;
  }

  // Symmetric matrix
  const M = [
    c00, c01, c02,
    c01, c11, c12,
    c02, c12, c22
  ];

  return dominantEigenvector3(M, 25).normalize();
}

export function preprocessGeometry(geometry) {
      // Keep a pristine copy for export paths that bake quaternions
  const originalGeometry = geometry.clone();
  const pos = geometry.attributes.position.array;
  const norm = geometry.attributes.normal.array;

  const triangles = [];
  let totalArea = 0;

  for (let i = 0; i < pos.length; i += 9) {
    const a = new THREE.Vector3(pos[i], pos[i + 1], pos[i + 2]);
    const b = new THREE.Vector3(pos[i + 3], pos[i + 4], pos[i + 5]);
    const c = new THREE.Vector3(pos[i + 6], pos[i + 7], pos[i + 8]);

    const n = new THREE.Vector3(
      norm[i] + norm[i + 3] + norm[i + 6],
      norm[i + 1] + norm[i + 4] + norm[i + 7],
      norm[i + 2] + norm[i + 5] + norm[i + 8]
    ).normalize();

    const area =
      new THREE.Vector3()
        .subVectors(b, a)
        .cross(new THREE.Vector3().subVectors(c, a))
        .length() * 0.5;

    totalArea += area;
    triangles.push({ a, b, c, normal: n, area });
  }

  geometry.computeBoundingBox();
  const bbox = geometry.boundingBox.clone();

  const principalAxis = estimatePrincipalAxis(triangles);

  return { triangles, bbox, totalArea, principalAxis, originalGeometry };
}
