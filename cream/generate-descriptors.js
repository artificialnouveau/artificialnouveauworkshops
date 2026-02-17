const faceapi = require("@vladmandic/face-api");
const { createCanvas, loadImage, Image, ImageData } = require("@napi-rs/canvas");
const fs = require("fs");
const path = require("path");

// Monkey-patch face-api with @napi-rs/canvas
faceapi.env.monkeyPatch({
  Canvas: createCanvas(1, 1).constructor,
  Image,
  ImageData,
  createCanvasElement: () => createCanvas(1, 1),
  createImageElement: () => new Image(),
});

const PHOTOS_DIR = "/Users/ahnjili_harmony/Documents/GitHub/cream/cream_photos";
const MODELS_DIR = path.join(__dirname, "models");
const OUTPUT_FILE = path.join(__dirname, "descriptors.json");

async function main() {
  console.log("Loading face-api.js models...");
  await faceapi.nets.ssdMobilenetv1.loadFromDisk(MODELS_DIR);
  await faceapi.nets.faceLandmark68Net.loadFromDisk(MODELS_DIR);
  await faceapi.nets.faceRecognitionNet.loadFromDisk(MODELS_DIR);
  console.log("Models loaded.");

  const files = fs
    .readdirSync(PHOTOS_DIR)
    .filter((f) => /\.(jpg|jpeg|png|webp)$/i.test(f))
    .sort();

  console.log(`Found ${files.length} image files.`);

  const results = [];
  let succeeded = 0;
  let failed = 0;

  for (let i = 0; i < files.length; i++) {
    const file = files[i];
    const label = path.basename(file, path.extname(file)).replace(/_/g, " ");
    const filePath = path.join(PHOTOS_DIR, file);

    try {
      const img = await loadImage(filePath);
      const cvs = createCanvas(img.width, img.height);
      const ctx = cvs.getContext("2d");
      ctx.drawImage(img, 0, 0);

      const detection = await faceapi
        .detectSingleFace(cvs)
        .withFaceLandmarks()
        .withFaceDescriptor();

      if (detection) {
        results.push({
          label,
          file,
          descriptor: Array.from(detection.descriptor),
        });
        succeeded++;
      } else {
        console.warn(`  [${i + 1}/${files.length}] No face detected: ${file}`);
        failed++;
      }
    } catch (err) {
      console.error(`  [${i + 1}/${files.length}] Error processing ${file}: ${err.message}`);
      failed++;
    }

    if ((i + 1) % 20 === 0 || i + 1 === files.length) {
      console.log(`  Processed ${i + 1}/${files.length} (${succeeded} OK, ${failed} failed)`);
    }
  }

  fs.writeFileSync(OUTPUT_FILE, JSON.stringify(results, null, 2));
  console.log(`\nDone! Saved ${results.length} descriptors to descriptors.json`);
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
