import http from "node:http";
import {readFile} from "node:fs/promises";
import path from "node:path";
import {fileURLToPath} from "node:url";
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const handlebars = process.argv[2];
if (!handlebars) throw new Error("Pass the absolute path to handlebars/dist/handlebars.js for the test harness.");
const types = {".mjs": "text/javascript", ".js": "text/javascript", ".css": "text/css", ".html": "text/html", ".hbs": "text/plain"};
http.createServer(async (request, response) => {
  try {
    const pathname = decodeURIComponent(new URL(request.url, "http://127.0.0.1").pathname);
    let file;
    if (pathname === "/handlebars.js") file = path.resolve(handlebars);
    else {
      const relative = pathname === "/" ? "tests/browser-harness.html" : pathname.replace(/^\/modules\//, "/").replace(/^\//, "");
      file = path.resolve(root, relative);
      if (!file.startsWith(root + path.sep)) throw new Error("Invalid path");
    }
    const content = await readFile(file);
    response.writeHead(200, {"Content-Type": types[path.extname(file)] ?? "application/octet-stream", "Cache-Control": "no-store"}); response.end(content);
  } catch { response.writeHead(404); response.end("Not found"); }
}).listen(43125, "127.0.0.1", () => process.stdout.write("Isolated harness: http://127.0.0.1:43125\n"));
