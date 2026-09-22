// Grouping faces by identity. Plain ES module with no DOM access, so the tests
// can import it directly under node.

/**
 * Distance below which two faces are treated as the same person.
 *
 * Embeddings are unit length, so this runs from 0 (identical) to 2 (opposite),
 * and it is not comparable with the value the previous model used.
 *
 * 0.9 sits between two measured populations. On a benchmark of one photo
 * degraded six ways — tilted 25 degrees, darkened, blurred, shrunk, and
 * compressed to nothing — the furthest pair of faces belonging to one person
 * was 0.579 apart and the nearest pair belonging to two people was 1.220. Any
 * value between those separates them perfectly; 0.9 is the middle.
 *
 * No single value suits every album. A set full of strangers wants something
 * tighter. So the app exposes a slider and the API takes `threshold` per
 * request.
 */
export const MATCH_THRESHOLD = 0.9;

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
 * distance is under the threshold and that cluster holds no other face from
 * the same photo; otherwise it starts a cluster of its own.
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
      // Two faces in the same photo are two different people. Nobody appears
      // twice in one frame, so this is free knowledge the descriptors do not
      // have, and it is what keeps everyone in a group shot from collapsing
      // into a single person.
      if (cluster.photos.has(face.fileName)) continue;

      const distance = euclidean(face.descriptor, cluster.centroid);
      if (distance < nearestDistance) {
        nearestDistance = distance;
        nearest = cluster;
      }
    }

    if (nearest && nearestDistance < threshold) {
      nearest.faces.push(face);
      nearest.photos.add(face.fileName);
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
        photos: new Set([face.fileName]),
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
