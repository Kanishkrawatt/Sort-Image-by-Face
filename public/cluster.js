// Grouping faces by identity. Plain ES module with no DOM access, so the tests
// can import it directly under node.

/**
 * Descriptor distance below which two faces are treated as the same person.
 * 0.55 is face-api's usual operating point; tune against real photos.
 */
export const MATCH_THRESHOLD = 0.55;

export function euclidean(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return Math.sqrt(sum);
}

/**
 * Group faces by identity, greedily.
 *
 * Each face joins the cluster whose centroid it is nearest to, provided that
 * distance is under the threshold; otherwise it starts a cluster of its own.
 *
 * A face carries the photo it came from, so a photo holding three people ends
 * up in three clusters. Returned clusters are sorted largest first.
 *
 * @param {Array<{fileName: string, descriptor: number[]}>} faces
 * @returns {Array<{id: number, centroid: number[], faces: object[]}>}
 */
export function clusterFaces(faces, threshold = MATCH_THRESHOLD) {
  const clusters = [];

  for (const face of faces) {
    let nearest = null;
    let nearestDistance = Infinity;

    for (const cluster of clusters) {
      const distance = euclidean(face.descriptor, cluster.centroid);
      if (distance < nearestDistance) {
        nearestDistance = distance;
        nearest = cluster;
      }
    }

    if (nearest && nearestDistance < threshold) {
      nearest.faces.push(face);
      // Fold the new descriptor into the running mean.
      const n = nearest.faces.length;
      for (let i = 0; i < nearest.centroid.length; i++) {
        nearest.centroid[i] += (face.descriptor[i] - nearest.centroid[i]) / n;
      }
    } else {
      clusters.push({
        id: clusters.length,
        centroid: Array.from(face.descriptor),
        faces: [face],
      });
    }
  }

  return clusters
    .sort((a, b) => b.faces.length - a.faces.length)
    .map((cluster, index) => ({ ...cluster, id: index }));
}

/** The distinct photos a cluster appears in, in the order first seen. */
export function photosIn(cluster) {
  return [...new Set(cluster.faces.map((face) => face.fileName))];
}
