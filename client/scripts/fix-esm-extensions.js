const fs = require("fs");
const path = require("path");

const targetFile = path.join(__dirname, "..", "dist", "index.js");

if (!fs.existsSync(targetFile)) {
  console.warn("fix-esm-extensions: dist/index.js not found, skipping.");
  process.exit(0);
}

const original = fs.readFileSync(targetFile, "utf8");
const updated = original.replace('export * from "./types";', 'export * from "./types.js";');

if (updated !== original) {
  fs.writeFileSync(targetFile, updated, "utf8");
}
