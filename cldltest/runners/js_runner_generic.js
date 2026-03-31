const fs = require("fs");
const path = require("path");
const ort = require("onnxruntime-node");

async function main() {
  if (process.argv.length !== 5) {
    console.log("Usage: node runners/run_node_onnx_generic.js <model_path> <input_json> <output_json>");
    process.exit(1);
  }

  const modelPath = path.resolve(process.argv[2]);
  const inputPath = path.resolve(process.argv[3]);
  const outputPath = path.resolve(process.argv[4]);

  const session = await ort.InferenceSession.create(modelPath);
  const raw = JSON.parse(fs.readFileSync(inputPath, "utf-8"));

  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];

  const results = [];

  for (let i = 0; i < raw.inputs.length; i++) {
    const item = raw.inputs[i];
    const shape = getShape(item);
    const flat = item.flat(Infinity);

    const tensor = new ort.Tensor("float32", Float32Array.from(flat), shape);

    const feeds = {};
    feeds[inputName] = tensor;

    const outputs = await session.run(feeds);
    const y = outputs[outputName];

    results.push({
      case_id: i,
      input: item,
      output: Array.from(y.data)
    });
  }

  fs.writeFileSync(outputPath, JSON.stringify(results, null, 2), "utf-8");
  console.log(`Node.js inference results saved to: ${outputPath}`);
  console.log(JSON.stringify(results, null, 2));
}

function getShape(arr) {
  const shape = [];
  let current = arr;
  while (Array.isArray(current)) {
    shape.push(current.length);
    current = current[0];
  }
  return shape;
}

main().catch((err) => {
  console.error("Node inference failed:", err);
  process.exit(1);
});