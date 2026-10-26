import http from "node:http";
import {readFile, mkdir, writeFile} from "node:fs/promises";
import path from "node:path";
import {fileURLToPath} from "node:url";
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const handlebars = process.argv[2];
const comparisonRoot = process.argv[3] ? path.resolve(process.argv[3]) : null;
if (!handlebars) throw new Error("Pass the absolute path to handlebars/dist/handlebars.js for the test harness.");
const types = {".mjs": "text/javascript", ".js": "text/javascript", ".css": "text/css", ".html": "text/html", ".hbs": "text/plain", ".wasm": "application/wasm", ".png": "image/png", ".json": "application/json"};
http.createServer(async (request, response) => {
  try {
    const pathname = decodeURIComponent(new URL(request.url, "http://127.0.0.1").pathname);
    if (request.method === "POST" && comparisonRoot && pathname.startsWith("/trial-results/")) {
      const name = pathname.slice("/trial-results/".length);
      if (!/^[a-z0-9-]+\.(png|json)$/.test(name) || request.headers.origin !== "http://127.0.0.1:43125") throw new Error("Invalid result");
      const chunks = []; let size = 0;
      for await (const chunk of request) {
        size += chunk.length;
        if (size > 10_000_000) throw new Error("Result too large");
        chunks.push(chunk);
      }
      const directory = path.join(comparisonRoot, "results"); await mkdir(directory, {recursive: true});
      await writeFile(path.join(directory, name), Buffer.concat(chunks));
      response.writeHead(201); response.end("Saved locally"); return;
    }
    if (request.method !== "GET") throw new Error("Unsupported method");
    let file;
    if (pathname === "/handlebars.js") file = path.resolve(handlebars);
    else if (comparisonRoot && pathname.startsWith("/comparison/")) {
      file = path.resolve(comparisonRoot, pathname.slice("/comparison/".length));
      if (!file.startsWith(comparisonRoot + path.sep)) throw new Error("Invalid comparison path");
    }
    else {
      const relative = pathname === "/" ? "tests/browser-harness.html" : pathname.replace(/^\/modules\//, "/").replace(/^\//, "");
      file = path.resolve(root, relative);
      if (!file.startsWith(root + path.sep)) throw new Error("Invalid path");
    }
    const content = await readFile(file);
    response.writeHead(200, {"Content-Type": types[path.extname(file)] ?? "application/octet-stream", "Cache-Control": "no-store"}); response.end(content);
  } catch { response.writeHead(404); response.end("Not found"); }
}).listen(43125, "127.0.0.1", () => process.stdout.write("Isolated harness: http://127.0.0.1:43125\n"));
