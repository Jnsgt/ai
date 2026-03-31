import json
import numpy as np
import onnxruntime as ort

def run_python_backend(model_path: str, input_json_path: str, output_json_path: str):
    with open(input_json_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    input_name = test_data["input_name"]
    shape = test_data["shape"]
    data = np.array(test_data["data"], dtype=np.float32).reshape(shape)

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    outputs = session.run(None, {input_name: data})
    output = outputs[0]

    result = {
        "runner": "python_onnxruntime",
        "model": model_path,
        "input_name": input_name,
        "input_shape": shape,
        "input_data": test_data["data"],
        "output_shape": list(output.shape),
        "output_data": output.reshape(-1).tolist(),
        "dtype": str(output.dtype),
        "status": "success"
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result