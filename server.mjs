import http from "node:http";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = __dirname;
const DATA_DIR = path.join(ROOT, "data");
const WEB_DIR = path.join(ROOT, "web");

const PORT = Number(process.env.PORT) || 8765;

/** @param {string | null | undefined} name */
function safeDataFilename(name) {
  if (!name || typeof name !== "string") return null;
  const base = path.basename(name);
  if (base !== name) return null;
  if (base.includes("..") || base.includes("/") || base.includes("\\")) return null;
  if (!/\.(tsv|txt)$/i.test(base)) return null;
  return base;
}

async function listDatasets() {
  try {
    const entries = await fs.readdir(DATA_DIR, { withFileTypes: true });
    return entries
      .filter((e) => e.isFile() && /\.(tsv|txt)$/i.test(e.name))
      .map((e) => e.name)
      .sort();
  } catch {
    return [];
  }
}

const NO_STORE = { "Cache-Control": "no-store, no-cache, must-revalidate", Pragma: "no-cache" };

/** @param {string} filePath */
function contentTypeFor(filePath) {
  const ext = path.extname(filePath).toLowerCase();
  if (ext === ".html") return "text/html; charset=utf-8";
  if (ext === ".js") return "text/javascript; charset=utf-8";
  if (ext === ".css") return "text/css; charset=utf-8";
  if (ext === ".json") return "application/json; charset=utf-8";
  if (ext === ".svg") return "image/svg+xml";
  return "application/octet-stream";
}

/**
 * @param {http.IncomingMessage} req
 * @returns {Promise<Buffer>}
 */
async function readBody(req) {
  const chunks = [];
  for await (const chunk of req) chunks.push(chunk);
  return Buffer.concat(chunks);
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url || "/", `http://127.0.0.1:${PORT}`);
  /** Trailing slashes break exact route matches; normalize for API + static paths. */
  const pathname = url.pathname.replace(/\/+$/, "") || "/";

  try {
    if (req.method === "GET" && pathname === "/api/health") {
      res.writeHead(200, {
        "Content-Type": "application/json; charset=utf-8",
        ...NO_STORE,
      });
      res.end(JSON.stringify({ ok: true, app: "mocap-label-app" }));
      return;
    }

    if (req.method === "GET" && pathname === "/api/datasets") {
      const names = await listDatasets();
      res.writeHead(200, { "Content-Type": "application/json; charset=utf-8", ...NO_STORE });
      res.end(JSON.stringify(names));
      return;
    }

    if (req.method === "GET" && pathname === "/api/dataset") {
      const raw = url.searchParams.get("name");
      const base = safeDataFilename(raw);
      if (!base) {
        res.writeHead(400, { "Content-Type": "text/plain" });
        res.end("Invalid name");
        return;
      }
      const allowed = await listDatasets();
      if (!allowed.includes(base)) {
        res.writeHead(404, { "Content-Type": "text/plain" });
        res.end("Not found");
        return;
      }
      const text = await fs.readFile(path.join(DATA_DIR, base), "utf8");
      res.writeHead(200, { "Content-Type": "text/plain; charset=utf-8" });
      res.end(text);
      return;
    }

    if (req.method === "GET" && pathname === "/api/skeleton") {
      const raw = url.searchParams.get("name");
      const base = safeDataFilename(raw);
      if (!base) {
        res.writeHead(400, { "Content-Type": "text/plain" });
        res.end("Invalid name");
        return;
      }
      const allowed = await listDatasets();
      if (!allowed.includes(base)) {
        res.writeHead(404, { "Content-Type": "text/plain" });
        res.end("Not found");
        return;
      }
      const stem = path.parse(base).name;
      const jsonPath = path.join(DATA_DIR, `${stem}_labelled_skeleton`, "mocap-graph.json");
      const text = await fs.readFile(jsonPath, "utf8").catch(() => null);
      if (text == null) {
        res.writeHead(404, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "no_skeleton", message: `Expected ${stem}_labelled_skeleton/mocap-graph.json` }));
        return;
      }
      res.writeHead(200, { "Content-Type": "application/json; charset=utf-8", ...NO_STORE });
      res.end(text);
      return;
    }

    if (req.method === "GET" && pathname === "/api/track-benchmark") {
      const raw = url.searchParams.get("name");
      const base = safeDataFilename(raw);
      if (!base) {
        res.writeHead(400, { "Content-Type": "application/json; charset=utf-8", ...NO_STORE });
        res.end(JSON.stringify({ ok: false, error: "Invalid or missing name" }));
        return;
      }
      const allowed = await listDatasets();
      if (!allowed.includes(base)) {
        res.writeHead(404, { "Content-Type": "application/json; charset=utf-8", ...NO_STORE });
        res.end(JSON.stringify({ ok: false, error: "Dataset not found" }));
        return;
      }
      const { runTrackBenchmark } = await import("./track-benchmark.mjs");
      const out = await runTrackBenchmark(base, { logProgress: false });
      res.writeHead(200, { "Content-Type": "application/json; charset=utf-8", ...NO_STORE });
      res.end(JSON.stringify(out));
      return;
    }

    if (req.method === "POST" && pathname === "/api/save-graph") {
      const buf = await readBody(req);
      let data;
      try {
        data = JSON.parse(buf.toString("utf8"));
      } catch {
        res.writeHead(400, { "Content-Type": "text/plain" });
        res.end("Invalid JSON");
        return;
      }
      const source = safeDataFilename(data.sourceFileName);
      if (!source || data.graph == null) {
        res.writeHead(400, { "Content-Type": "text/plain" });
        res.end("sourceFileName and graph required");
        return;
      }
      const allowed = await listDatasets();
      if (!allowed.includes(source)) {
        res.writeHead(403, { "Content-Type": "text/plain" });
        res.end("Unknown dataset");
        return;
      }
      const stem = path.parse(source).name;
      const outDir = path.join(DATA_DIR, `${stem}_labelled_skeleton`);
      await fs.mkdir(outDir, { recursive: true });
      const outPath = path.join(outDir, "mocap-graph.json");
      const json = JSON.stringify(data.graph, null, 2);
      await fs.writeFile(outPath, json, "utf8");
      const rel = path.join("data", `${stem}_labelled_skeleton`, "mocap-graph.json");
      res.writeHead(200, { "Content-Type": "application/json; charset=utf-8", ...NO_STORE });
      res.end(JSON.stringify({ ok: true, path: rel.replace(/\\/g, "/") }));
      return;
    }

    let rel = pathname === "/" ? "index.html" : pathname.slice(1);
    rel = path.normalize(rel).replace(/^(\.\.(\/|\\|$))+/, "");
    const filePath = path.join(WEB_DIR, rel);
    if (!filePath.startsWith(WEB_DIR)) {
      res.writeHead(403);
      res.end();
      return;
    }
    const stat = await fs.stat(filePath).catch(() => null);
    if (!stat || !stat.isFile()) {
      res.writeHead(404);
      res.end("Not found");
      return;
    }
    const body = await fs.readFile(filePath);
    res.writeHead(200, { "Content-Type": contentTypeFor(filePath) });
    res.end(body);
  } catch (err) {
    console.error(err);
    res.writeHead(500, { "Content-Type": "text/plain" });
    res.end("Server error");
  }
});

server.listen(PORT, () => {
  console.log(`MoCap labeller (Node): http://127.0.0.1:${PORT}/`);
  console.log(`Data: ${DATA_DIR}`);
  console.log("Use this server for /api — not python -m http.server");
});
