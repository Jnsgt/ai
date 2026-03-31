import json
import os
import sys
import numpy as np
import onnxruntime as ort


def main():
    if len(sys.argv) != 4:
        print("Usage: python runners/run_python_onnx_generic.py <model_path> <input_json> <output_json>")
        sys.exit(1)

    model_path = sys.argv[1]
    input_path = sys.argv[2]
    output_path = sys.argv[3]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

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