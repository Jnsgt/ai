const fs = require('fs');
const ort = require('onnxruntime-node');

async function main() {
  const modelPath = process.argv[2];
  const inputJsonPath = process.argv[3];
  const outputJsonPath = process.argv[4];

  if (!modelPath || !inputJsonPath || !outputJsonPath) {
    console.error("Usage: node js_runner.js <modelPath> <inputJsonPath> <outputJsonPath>");
    process.exit(1);
  }

  const raw = fs.readFileSync(inputJsonPath, 'utf-8');
  const testData = JSON.parse(raw);

  const inputName = testData.input_name;
  const shape = testData.shape;
  const data = new Float32Array(testData.data);

  const inputTensor = new ort.Tensor('float32', data, shape);
  const session = await ort.InferenceSession.create(modelPath);
  const outputs = await session.run({ [inputName]: inputTensor });
  const output = outputs.output;

  const resultJson = {
    runner: 'js_onnxruntime',
    model: modelPath,
    input_name: inputName,
    input_shape: shape,
    input_data: testData.data,
    output_shape: output.dims,
    output_data: Array.from(output.data),
    dtype: 'float32',
    status: 'success'
  };

  fs.writeFileSync(outputJsonPath, JSON.stringify(resultJson, null, 2), 'utf-8');
  console.log(`JS backend done: ${outputJsonPath}`);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});