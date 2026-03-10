import * as THREE from "three";

/**
 * Generate N roughly-even directions on a sphere using a Fibonacci spiral.
 * Each direction represents "down" in model space.
 */
export function fibonacciDirections(count = 42) {
  const dirs = [];
  const phi = Math.PI * (3 - Math.sqrt(5)); // golden angle

  for (let i = 0; i < count; i++) {
    const y = 1 - (i / (count - 1)) * 2; // +1 to -1
    const radius = Math.sqrt(Math.max(0, 1 - y * y));
    const theta = phi * i;

    const x = Math.cos(theta) * radius;
    const z = Math.sin(theta) * radius;

    dirs.push(new THREE.Vector3(x, y, z).normalize());
  }
  return dirs;
}

/**
 * Convert a "down direction" into a rotation that makes that direction align to -Z.
 * We rotate the model so that dir becomes the new "down".
 */
export function rotationFromDownDirection(dir) {
  const from = dir.clone().normalize();
  const to = new THREE.Vector3(0, 0, -1); // gravity down

  const q = new THREE.Quaternion().setFromUnitVectors(from, to);
  const m = new THREE.Matrix4().makeRotationFromQuaternion(q);

  return { quaternion: q, matrix: m, down: from };
}

/**
 * Build candidate orientations: list of { quaternion, matrix, down }.
 */
export function buildCandidates(count = 42) {
  return fibonacciDirections(count).map(rotationFromDownDirection);
}
