const fs = require("fs");
const path = require("path");
const ort = require("onnxruntime-node");

async function main() {
  const projectRoot = path.join(__dirname, "..");
  const modelPath = path.join(projectRoot, "models", "simple_regression.onnx");
  const inputPath = path.join(projectRoot, "tests", "sample_inputs.json");
  const outputPath = path.join(projectRoot, "reports", "node_output.json");

  const session = await ort.InferenceSession.create(modelPath);
  const raw = JSON.parse(fs.readFileSync(inputPath, "utf-8"));

  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];

  const results = [];

  for (let i = 0; i < raw.inputs.length; i++) {
    const item = raw.inputs[i];

    const flat = item.flat(Infinity);
    const tensor = new ort.Tensor("float32", Float32Array.from(flat), [1, 1]);

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

main().catch((err) => {
  console.error("Node inference failed:", err);
  process.exit(1);
});