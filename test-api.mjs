import { spawn } from "node:child_process";
import net from "node:net";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.dirname(fileURLToPath(import.meta.url));

function freePort() {
  return new Promise((resolve, reject) => {
    const s = net.createServer();
    s.listen(0, "127.0.0.1", () => {
      const addr = s.address();
      const p = typeof addr === "object" && addr ? addr.port : 0;
      s.close(() => resolve(p));
    });
    s.on("error", reject);
  });
}

async function waitForHealth(base, ms = 8000) {
  const t0 = Date.now();
  while (Date.now() - t0 < ms) {
    try {
      const r = await fetch(`${base}/api/health`, { cache: "no-store" });
      if (r.ok) {
        const j = await r.json();
        if (j && j.ok && j.app === "mocap-label-app") return;
      }
    } catch {
      /* retry */
    }
    await new Promise((r) => setTimeout(r, 40));
  }
  throw new Error("server did not become ready");
}

const port = await freePort();
const child = spawn(process.execPath, ["server.mjs"], {
  cwd: ROOT,
  env: { ...process.env, PORT: String(port) },
  stdio: ["ignore", "pipe", "pipe"],
});

let stderr = "";
child.stderr?.on("data", (c) => {
  stderr += c.toString();
});

const base = `http://127.0.0.1:${port}`;
let exit = 1;
try {
  await waitForHealth(base);

  const h = await fetch(`${base}/api/health`, {
    headers: { Accept: "application/json" },
    cache: "no-store",
  });
  if (!h.ok) throw new Error(`/api/health status ${h.status}`);
  const hj = await h.json();
  if (!hj.ok) throw new Error("/api/health body not ok");

  for (const path of ["/api/datasets", "/api/datasets/"]) {
    const r = await fetch(`${base}${path}`, {
      headers: { Accept: "application/json" },
      cache: "no-store",
    });
    if (!r.ok) throw new Error(`${path} status ${r.status}`);
    const ct = r.headers.get("content-type") || "";
    if (!ct.includes("application/json")) {
      throw new Error(`${path} content-type: ${ct}`);
    }
    const data = await r.json();
    if (!Array.isArray(data)) throw new Error(`${path} not a JSON array`);
  }

  console.log("test-api: ok", { port, datasets: "array" });
  exit = 0;
} catch (e) {
  console.error("test-api: fail", e?.message || e);
  if (stderr.trim()) console.error(stderr.trim());
  exit = 1;
} finally {
  child.kill("SIGTERM");
  await new Promise((r) => setTimeout(r, 200));
  if (child.exitCode === null) child.kill("SIGKILL");
}

process.exit(exit);
