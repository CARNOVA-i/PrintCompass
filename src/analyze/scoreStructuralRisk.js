import * as THREE from "three";
import { scoreOverhang } from "./scoreOverhang.js";

const WORLD_UP = new THREE.Vector3(0, 0, 1);
const EPSILON = 1e-6;
const NECK_BINS = 8;

const LOAD_TYPE_WEIGHTS = {
  compression: {
    loadAlignmentPenalty: 0.18,
    slendernessPenalty: 0.30,
    leverArmPenalty: 0.10,
    weakNeckPenalty: 0.14,
    topMassPenalty: 0.22,
    supportBurdenPenalty: 0.06,
  },
  bending: {
    loadAlignmentPenalty: 0.18,
    slendernessPenalty: 0.18,
    leverArmPenalty: 0.24,
    weakNeckPenalty: 0.22,
    topMassPenalty: 0.10,
    supportBurdenPenalty: 0.08,
  },
  shear: {
    loadAlignmentPenalty: 0.12,
    slendernessPenalty: 0.16,
    leverArmPenalty: 0.18,
    weakNeckPenalty: 0.28,
    topMassPenalty: 0.10,
    supportBurdenPenalty: 0.16,
  },
  tension: {
    loadAlignmentPenalty: 0.34,
    slendernessPenalty: 0.12,
    leverArmPenalty: 0.08,
    weakNeckPenalty: 0.28,
    topMassPenalty: 0.12,
    supportBurdenPenalty: 0.06,
  },
};

function clamp01(value) {
  return Math.max(0, Math.min(1, value));
}

function rotatedBounds(bbox, rotMatrix) {
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
  let minY = Infinity;
  let minZ = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let maxZ = -Infinity;

  for (const point of corners) {
    point.applyMatrix4(rotMatrix);
    minX = Math.min(minX, point.x);
    minY = Math.min(minY, point.y);
    minZ = Math.min(minZ, point.z);
    maxX = Math.max(maxX, point.x);
    maxY = Math.max(maxY, point.y);
    maxZ = Math.max(maxZ, point.z);
  }

  return {
    minX,
    minY,
    minZ,
    maxX,
    maxY,
    maxZ,
    width: maxX - minX,
    depth: maxY - minY,
    height: maxZ - minZ,
    centerX: (minX + maxX) * 0.5,
    centerY: (minY + maxY) * 0.5,
  };
}

function computeBottomContactArea(triangles, rotMatrix, minZ, height) {
  const centroid = new THREE.Vector3();
  const normal = new THREE.Vector3();
  const tolerance = Math.max(0.5, height * 0.02);

  let contactArea = 0;

  for (const triangle of triangles) {
    centroid
      .copy(triangle.a)
      .add(triangle.b)
      .add(triangle.c)
      .multiplyScalar(1 / 3)
      .applyMatrix4(rotMatrix);

    normal.copy(triangle.normal).transformDirection(rotMatrix);

    const nearBed = centroid.z - minZ <= tolerance;
    const facesBed = normal.dot(WORLD_UP) <= -0.85;

    if (nearBed && facesBed) {
      contactArea += triangle.area;
    }
  }

  return contactArea;
}

function computeMassDistribution(triangles, rotMatrix, bounds) {
  const centroid = new THREE.Vector3();
  const bandAreas = new Array(NECK_BINS).fill(0);

  let totalArea = 0;
  let weightedX = 0;
  let weightedY = 0;
  let weightedZ = 0;
  let upperArea = 0;

  for (const triangle of triangles) {
    centroid
      .copy(triangle.a)
      .add(triangle.b)
      .add(triangle.c)
      .multiplyScalar(1 / 3)
      .applyMatrix4(rotMatrix);

    const area = triangle.area;
    const zNorm = clamp01((centroid.z - bounds.minZ) / Math.max(bounds.height, EPSILON));
    const binIndex = Math.min(NECK_BINS - 1, Math.floor(zNorm * NECK_BINS));

    bandAreas[binIndex] += area;
    totalArea += area;
    weightedX += centroid.x * area;
    weightedY += centroid.y * area;
    weightedZ += centroid.z * area;

    if (zNorm >= 0.55) {
      upperArea += area;
    }
  }

  return {
    totalArea,
    centroidX: weightedX / Math.max(totalArea, EPSILON),
    centroidY: weightedY / Math.max(totalArea, EPSILON),
    centroidZ: weightedZ / Math.max(totalArea, EPSILON),
    upperAreaFraction: upperArea / Math.max(totalArea, EPSILON),
    bandAreas,
  };
}

function computeWeakNeckPenalty(bandAreas, upperAreaFraction) {
  const maxBandArea = Math.max(...bandAreas, EPSILON);
  const neckBands = bandAreas.slice(2, NECK_BINS - 1);
  const neckArea = Math.min(...neckBands, maxBandArea);
  const neckRatio = neckArea / maxBandArea;

  return clamp01((1 - neckRatio) * (0.35 + (0.65 * upperAreaFraction)));
}

function getRiskLabel(score) {
  if (score <= 0.28) return "Low structural risk";
  if (score <= 0.55) return "Moderate structural risk";
  return "High structural risk";
}

function normalizeLoadType(loadType) {
  switch ((loadType ?? "").trim().toLowerCase()) {
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

/**
 * Heuristic structural-risk score for a candidate orientation.
 * Returns a normalized score where 0 is lower risk and 1 is higher risk.
 */
export function scoreStructuralRisk(
  analysisData,
  rotMatrix,
  rotQuat,
  loadAxisModelSpace = null,
  loadType = "compression"
) {
  const { bbox, triangles, principalAxis, totalArea } = analysisData;
  const bounds = rotatedBounds(bbox, rotMatrix);
  const footprint = Math.max(bounds.width * bounds.depth, EPSILON);
  const footprintRadius = Math.max(Math.sqrt(footprint) * 0.5, EPSILON);
  const loadAxis = (loadAxisModelSpace ?? principalAxis).clone().normalize();
  const normalizedLoadType = normalizeLoadType(loadType);
  const weights = LOAD_TYPE_WEIGHTS[normalizedLoadType];
  const loadRot = loadAxis.applyQuaternion(rotQuat).normalize();
  const mass = computeMassDistribution(triangles, rotMatrix, bounds);
  const baseContactArea = computeBottomContactArea(triangles, rotMatrix, bounds.minZ, bounds.height);
  const weightedOverhang = scoreOverhang(triangles, rotMatrix);

  const loadAlignmentPenalty = Math.abs(loadRot.z);
  const slenderness = bounds.height / Math.max(Math.sqrt(footprint), EPSILON);
  const slendernessPenalty = clamp01((slenderness - 1.1) / 2.4);

  const centroidOffset = Math.hypot(
    mass.centroidX - bounds.centerX,
    mass.centroidY - bounds.centerY
  );
  const leverArmPenalty = clamp01(
    (centroidOffset / footprintRadius) * (0.35 + (0.65 * mass.upperAreaFraction))
  );

  const weakNeckPenalty = computeWeakNeckPenalty(mass.bandAreas, mass.upperAreaFraction);
  const baseContactRatio = baseContactArea / Math.max(totalArea, EPSILON);
  const topMassPenalty = clamp01(
    mass.upperAreaFraction * (1 - clamp01(baseContactRatio / 0.18))
  );
  const supportBurdenPenalty = clamp01((weightedOverhang / Math.max(totalArea, EPSILON)) / 0.3);

  const normalizedScore =
    (weights.loadAlignmentPenalty * loadAlignmentPenalty) +
    (weights.slendernessPenalty * slendernessPenalty) +
    (weights.leverArmPenalty * leverArmPenalty) +
    (weights.weakNeckPenalty * weakNeckPenalty) +
    (weights.topMassPenalty * topMassPenalty) +
    (weights.supportBurdenPenalty * supportBurdenPenalty);

  return {
    normalizedScore: clamp01(normalizedScore),
    label: getRiskLabel(normalizedScore),
    loadType: normalizedLoadType,
    breakdown: {
      loadAlignmentPenalty,
      slendernessPenalty,
      leverArmPenalty,
      weakNeckPenalty,
      topMassPenalty,
      supportBurdenPenalty,
    },
  };
}
