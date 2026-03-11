import * as THREE from "three";
import { scoreOverhang } from "./scoreOverhang.js";

const WORLD_UP = new THREE.Vector3(0, 0, 1);
const EPSILON = 1e-6;

function clamp01(value) {
  return Math.max(0, Math.min(1, value));
}

function inverseLerp(min, max, value) {
  if (!Number.isFinite(min) || !Number.isFinite(max) || Math.abs(max - min) < EPSILON) {
    return 0;
  }
  return clamp01((value - min) / (max - min));
}

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

  let minX = Infinity;
  let minY = Infinity;
  let minZ = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let maxZ = -Infinity;

  for (const corner of corners) {
    corner.applyMatrix4(rotMatrix);
    minX = Math.min(minX, corner.x);
    minY = Math.min(minY, corner.y);
    minZ = Math.min(minZ, corner.z);
    maxX = Math.max(maxX, corner.x);
    maxY = Math.max(maxY, corner.y);
    maxZ = Math.max(maxZ, corner.z);
  }

  return {
    minZ,
    width: maxX - minX,
    depth: maxY - minY,
    height: maxZ - minZ,
  };
}

function computeBottomContactArea(triangles, rotMatrix, minZ, height) {
  const normal = new THREE.Vector3();
  const centroid = new THREE.Vector3();
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

function createCandidateMetrics(analysisData, candidate) {
  const extents = rotatedExtents(analysisData.bbox, candidate.matrix);
  const footprint = Math.max(EPSILON, extents.width * extents.depth);
  const supportNeed = scoreOverhang(analysisData.triangles, candidate.matrix) / Math.max(analysisData.totalArea, EPSILON);
  const bottomContactArea = computeBottomContactArea(
    analysisData.triangles,
    candidate.matrix,
    extents.minZ,
    extents.height
  );
  const contactRatio = bottomContactArea / Math.max(analysisData.totalArea, EPSILON);
  const slenderness = extents.height / Math.sqrt(footprint);
  const bedStability = clamp01((0.65 * contactRatio) + (0.35 * (1 / (1 + slenderness))));

  return {
    xDeg: candidate.xDeg,
    yDeg: candidate.yDeg,
    quaternion: candidate.quaternion.clone(),
    matrix: candidate.matrix.clone(),
    metrics: {
      supportNeed,
      printHeight: extents.height,
      bedStability,
      footprint,
      contactRatio,
      slenderness,
    },
  };
}

function buildSampleCandidates(stepDegrees = 45) {
  const candidates = [];

  for (let xDeg = -180; xDeg < 180; xDeg += stepDegrees) {
    for (let yDeg = -180; yDeg < 180; yDeg += stepDegrees) {
      const euler = new THREE.Euler(
        THREE.MathUtils.degToRad(xDeg),
        THREE.MathUtils.degToRad(yDeg),
        0,
        "XYZ"
      );
      const quaternion = new THREE.Quaternion().setFromEuler(euler);
      const matrix = new THREE.Matrix4().makeRotationFromQuaternion(quaternion);

      candidates.push({ xDeg, yDeg, quaternion, matrix });
    }
  }

  return candidates;
}

function scoreCandidates(candidates) {
  const supportValues = candidates.map((candidate) => candidate.metrics.supportNeed);
  const heightValues = candidates.map((candidate) => candidate.metrics.printHeight);
  const stabilityValues = candidates.map((candidate) => candidate.metrics.bedStability);

  const supportMin = Math.min(...supportValues);
  const supportMax = Math.max(...supportValues);
  const heightMin = Math.min(...heightValues);
  const heightMax = Math.max(...heightValues);
  const stabilityMin = Math.min(...stabilityValues);
  const stabilityMax = Math.max(...stabilityValues);

  return candidates.map((candidate) => {
    const normalizedSupport = inverseLerp(supportMin, supportMax, candidate.metrics.supportNeed);
    const normalizedHeight = inverseLerp(heightMin, heightMax, candidate.metrics.printHeight);
    const normalizedStability = inverseLerp(stabilityMin, stabilityMax, candidate.metrics.bedStability);
    const normalizedStabilityPenalty = 1 - normalizedStability;

    const balancedScore =
      (0.45 * normalizedSupport) +
      (0.35 * normalizedStabilityPenalty) +
      (0.20 * normalizedHeight);

    return {
      ...candidate,
      normalized: {
        supportNeed: normalizedSupport,
        printHeight: normalizedHeight,
        stabilityPenalty: normalizedStabilityPenalty,
      },
      scores: {
        balanced: balancedScore,
        support: normalizedSupport,
        stability: normalizedStabilityPenalty,
      },
    };
  });
}

function pickRecommendation(sorted, usedKeys) {
  for (const candidate of sorted) {
    const key = `${candidate.xDeg}:${candidate.yDeg}`;
    if (usedKeys.has(key)) continue;
    usedKeys.add(key);
    return candidate;
  }

  return sorted[0] ?? null;
}

function formatRecommendation(label, candidate) {
  if (!candidate) return null;

  return {
    label,
    xDeg: candidate.xDeg,
    yDeg: candidate.yDeg,
    quaternion: candidate.quaternion.clone(),
    metrics: candidate.metrics,
    normalized: candidate.normalized,
    scores: candidate.scores,
  };
}

export function analyzeOrientationRecommendations(analysisData, options = {}) {
  const stepDegrees = options.stepDegrees ?? 45;
  const sampledCandidates = buildSampleCandidates(stepDegrees).map((candidate) =>
    createCandidateMetrics(analysisData, candidate)
  );
  const scoredCandidates = scoreCandidates(sampledCandidates);

  const usedKeys = new Set();
  const bestBalanced = pickRecommendation(
    [...scoredCandidates].sort((a, b) => a.scores.balanced - b.scores.balanced),
    usedKeys
  );
  const lowestSupport = pickRecommendation(
    [...scoredCandidates].sort((a, b) => a.scores.support - b.scores.support),
    usedKeys
  );
  const bestStability = pickRecommendation(
    [...scoredCandidates].sort((a, b) => a.scores.stability - b.scores.stability),
    usedKeys
  );

  return {
    stepDegrees,
    candidateCount: scoredCandidates.length,
    recommendations: [
      formatRecommendation("Best Balanced", bestBalanced),
      formatRecommendation("Lowest Support", lowestSupport),
      formatRecommendation("Best Stability", bestStability),
    ].filter(Boolean),
  };
}
