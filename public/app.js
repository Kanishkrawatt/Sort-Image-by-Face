import { clusterFaces, photosIn } from "./cluster.js";
import { activeBackend, detectFaces, init } from "./detect.js";

/** Photos decoded at once. Enough to keep the GPU fed during file reads. */
const CONCURRENCY = 3;

const state = {
  files: new Map(), // path -> File
  faces: [],
  noFaces: [],
  skipped: [],
  clusters: [],
  names: new Map(), // cluster id -> chosen name
};

const el = {
  picker: document.querySelector("#picker"),
  pick: document.querySelector("#pick"),
  status: document.querySelector("#status"),
  progress: document.querySelector("#progress"),
  bar: document.querySelector("#bar"),
  results: document.querySelector("#results"),
  summary: document.querySelector("#summary"),
};

function setStatus(text) {
  el.status.textContent = text;
}

function setProgress(done, total) {
  el.progress.hidden = false;
  el.bar.style.width = `${Math.round((done / total) * 100)}%`;
  setStatus(`Looking at photo ${done} of ${total}`);
}

/** Run `task` over `items`, at most `limit` at a time. */
async function mapLimited(items, limit, task) {
  let cursor = 0;
  const runners = Array.from(
    { length: Math.min(limit, items.length) },
    async () => {
      while (cursor < items.length) {
        const index = cursor++;
        await task(items[index], index);
      }
    },
  );
  await Promise.all(runners);
}

async function run(fileList) {
  const files = [...fileList].filter((file) => file.type.startsWith("image/"));
  if (files.length === 0) {
    setStatus("No images in that folder.");
    return;
  }

  Object.assign(state, {
    files: new Map(),
    faces: [],
    noFaces: [],
    skipped: [],
    clusters: [],
    names: new Map(),
  });
  el.results.replaceChildren();
  el.pick.disabled = true;

  try {
    await init(setStatus);
    setStatus(`Ready (${activeBackend()})`);

    let done = 0;
    await mapLimited(files, CONCURRENCY, async (file) => {
      const path = file.webkitRelativePath || file.name;
      state.files.set(path, file);

      try {
        const found = await detectFaces(file);
        if (found.length === 0) {
          state.noFaces.push(path);
        } else {
          for (const face of found) {
            state.faces.push({ ...face, fileName: path });
          }
        }
      } catch (error) {
        // One bad photo must not end the run.
        console.warn("Skipped", path, error);
        state.skipped.push(path);
      }

      setProgress(++done, files.length);
    });

    setStatus("Grouping faces");
    // Yield once so the status paints before clustering blocks.
    await new Promise((resolve) => setTimeout(resolve, 0));
    state.clusters = clusterFaces(state.faces);

    render();
    setStatus(
      `${state.clusters.length} ${state.clusters.length === 1 ? "person" : "people"} across ${files.length} photos`,
    );
  } catch (error) {
    console.error(error);
    setStatus(`Something went wrong: ${error.message}`);
  } finally {
    el.pick.disabled = false;
    el.progress.hidden = true;
  }
}

function basename(path) {
  return path.slice(path.lastIndexOf("/") + 1);
}

function photoTile(path) {
  const img = document.createElement("img");
  img.className = "photo";
  img.loading = "lazy";
  img.alt = basename(path);
  img.title = path;
  img.src = URL.createObjectURL(state.files.get(path));
  return img;
}

async function download(cluster) {
  const paths = photosIn(cluster);
  const zip = new JSZip();
  for (const path of paths) zip.file(path, state.files.get(path));

  // Photos are already compressed; deflating them again only costs time.
  const blob = await zip.generateAsync({ type: "blob", compression: "STORE" });
  const label = state.names.get(cluster.id) || `person-${cluster.id + 1}`;

  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `${label}.zip`;
  link.click();
  URL.revokeObjectURL(link.href);
}

function personCard(cluster) {
  const paths = photosIn(cluster);
  const card = document.createElement("section");
  card.className = "person";

  const head = document.createElement("header");

  const avatar = document.createElement("img");
  avatar.className = "avatar";
  avatar.src = cluster.faces[0].thumbnail;
  avatar.alt = "";

  const name = document.createElement("input");
  name.className = "name";
  name.value = state.names.get(cluster.id) || `Person ${cluster.id + 1}`;
  name.setAttribute("aria-label", "Name this person");
  name.addEventListener("change", () => {
    state.names.set(cluster.id, name.value.trim());
  });

  const count = document.createElement("span");
  count.className = "count";
  count.textContent = `${paths.length} ${paths.length === 1 ? "photo" : "photos"}`;

  const save = document.createElement("button");
  save.textContent = "Download zip";
  save.addEventListener("click", async () => {
    save.disabled = true;
    save.textContent = "Zipping…";
    try {
      await download(cluster);
    } finally {
      save.disabled = false;
      save.textContent = "Download zip";
    }
  });

  head.append(avatar, name, count, save);

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

function render() {
  const nodes = state.clusters.map(personCard);

  if (state.noFaces.length) {
    nodes.push(plainSection("No faces found", state.noFaces));
  }

  el.summary.textContent = state.skipped.length
    ? `${state.skipped.length} file(s) could not be read and were skipped.`
    : "";

  el.results.replaceChildren(...nodes);
}

el.pick.addEventListener("click", () => el.picker.click());
el.picker.addEventListener("change", (event) => run(event.target.files));
