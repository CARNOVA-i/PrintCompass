import * as THREE from "three";

/**
 * Overhang proxy:
 * Sum of triangle area weighted by how "downward" the face normal points.
 *
 * If rotated normal points strongly downward (nz near -1) => big support pain.
 * If normal points upward (nz near +1) => no support.
 */
export function scoreOverhang(triangles, rotMatrix) {
  const n = new THREE.Vector3();
  let weightedDownArea = 0;

  for (const t of triangles) {
    n.copy(t.normal).applyMatrix4(rotMatrix).normalize();

    // Downwardness: 0..1 where 1 is fully downward-facing
    const down = Math.max(0, -n.z);

    // Power curve: emphasize the truly bad faces
    weightedDownArea += t.area * (down * down);
  }

  return weightedDownArea;
}
    