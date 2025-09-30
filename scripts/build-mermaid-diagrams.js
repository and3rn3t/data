const fs = require("fs");
const path = require("path");
const { execSync } = require("child_process");

/**
 * Build all Mermaid diagrams to PNG images
 */

const diagramsDir = "docs/diagrams";
const imagesDir = "docs/images";
const mmdcPath = "npx @mermaid-js/mermaid-cli";

// Ensure images directory exists
if (!fs.existsSync(imagesDir)) {
  fs.mkdirSync(imagesDir, { recursive: true });
}

// Configuration for mermaid-cli
const config = {
  theme: "default",
  backgroundColor: "white",
  width: 1200,
  height: 800,
  scale: 2,
};

// Write config file
fs.writeFileSync("mermaid-config.json", JSON.stringify(config, null, 2));

// Get all .mmd files
const mmdFiles = fs
  .readdirSync(diagramsDir)
  .filter((file) => file.endsWith(".mmd"));

console.log("🎨 Building Mermaid diagrams...\n");

mmdFiles.forEach((file) => {
  const inputPath = path.join(diagramsDir, file);
  const outputName = file.replace(".mmd", ".png");
  const outputPath = path.join(imagesDir, outputName);

  try {
    console.log(`📊 Building ${file} -> ${outputName}`);

    execSync(
      `${mmdcPath} -i "${inputPath}" -o "${outputPath}" -c mermaid-config.json -b white`,
      {
        stdio: "pipe",
      }
    );

    console.log(`✅ Successfully built ${outputName}`);
  } catch (error) {
    console.error(`❌ Failed to build ${file}:`, error.message);
  }
});

// Clean up config file
fs.unlinkSync("mermaid-config.json");

console.log("\n🎉 Diagram build complete!");
console.log(`\nGenerated images in ${imagesDir}/:`);
fs.readdirSync(imagesDir)
  .filter((file) => file.endsWith(".png"))
  .forEach((file) => console.log(`  📈 ${file}`));
