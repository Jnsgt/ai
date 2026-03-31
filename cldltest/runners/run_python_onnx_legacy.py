import json
import os
import numpy as np
import onnxruntime as ort


def main():
    project_root = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(project_root, "models", "simple_regression.onnx")
    input_path = os.path.join(project_root, "tests", "sample_inputs.json")
    output_path = os.path.join(project_root, "reports", "python_output.json")

    os.makedirs(os.path.join(project_root, "reports"), exist_ok=True)

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []

    for idx, item in enumerate(data["inputs"]):
        x = np.array(item, dtype=np.float32)
        y = session.run([output_name], {input_name: x})[0]
        results.append({
            "case_id": idx,
            "input": x.tolist(),
            "output": y.tolist()
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Python inference results saved to: {output_path}")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()