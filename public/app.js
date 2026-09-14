import { clusterFaces, MATCH_THRESHOLD } from "./cluster.js";
import { activeBackend, detectFaces, init } from "./detect.js";

/** Photos decoded at once. Enough to keep the GPU fed during file reads. */
const CONCURRENCY = 3;

const state = {
  files: new Map(), // path -> File
  previews: new Map(), // path -> small JPEG data URL
  faces: [], // every face found, each with a stable uid
  noFaces: [],
  skipped: [],
  names: new Map(), // face uid -> the name typed for its group
  merges: [], // [a, b] pairs of group indexes the user joined
  selected: new Set(),
  threshold: MATCH_THRESHOLD,
};

const el = {
  picker: document.querySelector("#picker"),
  pick: document.querySelector("#pick"),
  status: document.querySelector("#status"),
  progress: document.querySelector("#progress"),
  bar: document.querySelector("#bar"),
  results: document.querySelector("#results"),
  summary: document.querySelector("#summary"),
  tuning: document.querySelector("#tuning"),
  threshold: document.querySelector("#threshold"),
  thresholdValue: document.querySelector("#thresholdValue"),
  merge: document.querySelector("#merge"),
  reset: document.querySelector("#reset"),
};

const setStatus = (text) => { el.status.textContent = text; };

function setProgress(done, total) {
  el.progress.hidden = false;
  el.bar.style.width = `${Math.round((done / total) * 100)}%`;
  setStatus(`Looking at photo ${done} of ${total}`);
}

/** Run `task` over `items`, at most `limit` at a time. */
async function mapLimited(items, limit, task) {
  let cursor = 0;
  const runners = Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (cursor < items.length) await task(items[cursor++]);
  });
  await Promise.all(runners);
}

async function run(fileList) {
  const files = [...fileList].filter((file) => file.type.startsWith("image/"));
  if (files.length === 0) {
    setStatus("No images in that folder.");
    return;
  }

  Object.assign(state, {
    files: new Map(), previews: new Map(), faces: [], noFaces: [], skipped: [],
    names: new Map(), merges: [], selected: new Set(),
  });
  el.results.replaceChildren();
  el.tuning.hidden = true;
  el.pick.disabled = true;

  try {
    await init(setStatus);
    setStatus(`Ready (${activeBackend()})`);

    let done = 0;
    await mapLimited(files, CONCURRENCY, async (file) => {
      const path = file.webkitRelativePath || file.name;
      state.files.set(path, file);

      try {
        const { faces, preview } = await detectFaces(file);
        state.previews.set(path, preview);
        if (faces.length === 0) {
          state.noFaces.push(path);
        } else {
          faces.forEach((face, index) => {
            // A stable identity for this face, so a name survives re-grouping.
            state.faces.push({ ...face, fileName: path, uid: `${path}#${index}` });
          });
        }
      } catch (error) {
        console.warn("Skipped", path, error);
        state.skipped.push(path);
      }

      setProgress(++done, files.length);
    });

    setStatus("Grouping faces");
    await new Promise((resolve) => setTimeout(resolve, 0));
    el.tuning.hidden = false;
    render();
  } catch (error) {
    console.error(error);
    setStatus(`Something went wrong: ${error.message}`);
  } finally {
    el.pick.disabled = false;
    el.progress.hidden = true;
  }
}

/**
 * Cluster at the current threshold, then apply the merges the user asked for.
 * Descriptors are already in memory, so this is instant — no photo is read
 * again when the threshold moves.
 */
function groups() {
  const clusters = clusterFaces(
    state.faces.map((face) => ({ ...face })),
    state.threshold,
  );

  const parent = clusters.map((_, i) => i);
  const find = (i) => (parent[i] === i ? i : (parent[i] = find(parent[i])));
  for (const [a, b] of state.merges) {
    if (a >= clusters.length || b >= clusters.length) continue;
    const rootA = find(a);
    const rootB = find(b);
    if (rootA !== rootB) parent[rootB] = rootA;
  }

  const byRoot = new Map();
  clusters.forEach((cluster, i) => {
    const root = find(i);
    if (!byRoot.has(root)) byRoot.set(root, []);
    byRoot.get(root).push(...cluster.faces);
  });

  return [...byRoot.values()]
    .sort((a, b) => b.length - a.length)
    .map((faces, id) => ({ id, faces }));
}

const basename = (path) => path.slice(path.lastIndexOf("/") + 1);
const photosOf = (group) => [...new Set(group.faces.map((f) => f.fileName))];

/** The name the user typed for this group, found via any face it contains. */
function nameOf(group) {
  for (const face of group.faces) {
    const chosen = state.names.get(face.uid);
    if (chosen) return chosen;
  }
  return `Person ${group.id + 1}`;
}

function photoTile(path) {
  const img = document.createElement("img");
  img.className = "photo";
  img.loading = "lazy";
  img.alt = basename(path);
  img.title = path;
  // The small preview made during detection, not the original: decoding a
  // 12MP file for a 110px tile is what makes a large album crawl.
  img.src = state.previews.get(path);
  return img;
}

async function download(group) {
  const zip = new JSZip();
  for (const path of photosOf(group)) zip.file(path, state.files.get(path));

  // Photos are already compressed; deflating them again only costs time.
  const blob = await zip.generateAsync({ type: "blob", compression: "STORE" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `${nameOf(group).replace(/[^\w -]/g, "") || "person"}.zip`;
  link.click();
  URL.revokeObjectURL(link.href);
}

function personCard(group) {
  const paths = photosOf(group);
  const card = document.createElement("section");
  card.className = "person";

  const head = document.createElement("header");

  const tick = document.createElement("input");
  tick.type = "checkbox";
  tick.className = "tick";
  tick.checked = state.selected.has(group.id);
  tick.title = "Select to merge with another group";
  tick.addEventListener("change", () => {
    if (tick.checked) state.selected.add(group.id);
    else state.selected.delete(group.id);
    updateMergeButton();
  });

  const avatar = document.createElement("img");
  avatar.className = "avatar";
  avatar.src = group.faces[0].thumbnail;
  avatar.alt = "";

  const name = document.createElement("input");
  name.className = "name";
  name.value = nameOf(group);
  name.setAttribute("aria-label", "Name this person");
  name.addEventListener("change", () => {
    // Store against every face, so the name follows them through re-grouping.
    for (const face of group.faces) state.names.set(face.uid, name.value.trim());
  });

  const count = document.createElement("span");
  count.className = "count";
  count.textContent = `${paths.length} ${paths.length === 1 ? "photo" : "photos"}, ${group.faces.length} ${group.faces.length === 1 ? "face" : "faces"}`;

  const save = document.createElement("button");
  save.textContent = "Download zip";
  save.addEventListener("click", async () => {
    save.disabled = true;
    save.textContent = "Zipping…";
    try {
      await download(group);
    } finally {
      save.disabled = false;
      save.textContent = "Download zip";
    }
  });

  head.append(tick, avatar, name, count, save);

  const grid = document.createElement("div");
  grid.className = "grid";
  grid.append(...paths.map(photoTile));

  card.append(head, grid);
  return card;
}

function plainSection(title, paths) {
  const card = document.createElement("section");
  card.className = "person muted";
  const head = document.createElement("header");
  const label = document.createElement("strong");
  label.textContent = `${title} (${paths.length})`;
  head.append(label);
  const grid = document.createElement("div");
  grid.className = "grid";
  grid.append(...paths.map(photoTile));
  card.append(head, grid);
  return card;
}

function updateMergeButton() {
  el.merge.disabled = state.selected.size < 2;
  el.merge.textContent =
    state.selected.size < 2 ? "Merge selected" : `Merge ${state.selected.size} selected`;
}

function render() {
  const people = groups();
  const nodes = people.map(personCard);

  if (state.noFaces.length) nodes.push(plainSection("No faces found", state.noFaces));

  const notes = [];
  if (state.skipped.length) notes.push(`${state.skipped.length} file(s) could not be read`);
  if (state.merges.length) notes.push(`${state.merges.length} manual merge(s)`);
  el.summary.textContent = notes.join(" · ");

  el.results.replaceChildren(...nodes);
  el.thresholdValue.textContent = state.threshold.toFixed(2);
  el.reset.disabled = state.merges.length === 0;
  updateMergeButton();

  setStatus(
    `${people.length} ${people.length === 1 ? "person" : "people"} across ${state.files.size} photos`,
  );
}

el.pick.addEventListener("click", () => el.picker.click());
el.picker.addEventListener("change", (event) => run(event.target.files));

el.threshold.addEventListener("input", () => {
  state.threshold = Number(el.threshold.value);
  // Group numbers change with the threshold, so the merges recorded against
  // the old numbering no longer mean anything.
  state.merges = [];
  state.selected.clear();
  render();
});

el.merge.addEventListener("click", () => {
  const [first, ...rest] = [...state.selected];
  for (const other of rest) state.merges.push([first, other]);
  state.selected.clear();
  render();
});

el.reset.addEventListener("click", () => {
  state.merges = [];
  state.selected.clear();
  render();
});
