import * as THREE from "three";
import { scoreOverhang } from "./scoreOverhang.js";

function rotatedExtents(bbox, rotMatrix) {
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

  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

  for (const p of corners) {
    p.applyMatrix4(rotMatrix);
    minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x);
    minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y);
    minZ = Math.min(minZ, p.z); maxZ = Math.max(maxZ, p.z);
  }

  return { width: maxX - minX, depth: maxY - minY, height: maxZ - minZ };
}

/**
 * Strength score (lower is better):
 * - LoadUpPenalty: keep the chosen load direction OUT of Z (layer weakness).
 * - Slenderness: avoid tall skinny prints.
 * - Small overhang term: avoid insane support explosions.
 *
 * loadAxisModelSpace: THREE.Vector3 in MODEL space (before rotation).
 * If null, we fall back to principalAxis.
 */
export function scoreStrength(analysisData, rotMatrix, rotQuat, loadAxisModelSpace = null) {
  const { bbox, triangles, principalAxis } = analysisData;

  const loadAxis = (loadAxisModelSpace ?? principalAxis).clone().normalize();

  // Rotate the load axis into world after applying candidate rotation
  const loadRot = loadAxis.applyQuaternion(rotQuat).normalize();

  // 0 good (load mostly in XY), 1 bad (load vertical along Z)
  const loadUpPenalty = Math.abs(loadRot.z);

  // Slenderness from rotated bbox
  const ex = rotatedExtents(bbox, rotMatrix);
  const footprint = Math.max(1e-9, ex.width * ex.depth);
  const slenderness = ex.height / Math.sqrt(footprint);

  // Small support sanity check
  const overhang = scoreOverhang(triangles, rotMatrix);

  return (0.60 * loadUpPenalty) + (0.32 * slenderness) + (0.08 * (overhang / 1e5));
}
