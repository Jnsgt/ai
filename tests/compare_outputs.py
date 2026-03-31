import json
import sys
import numpy as np


def flatten_to_numpy(x):
    return np.array(x, dtype=np.float32).reshape(-1)


def max_abs_diff(a, b):
    a = flatten_to_numpy(a)
    b = flatten_to_numpy(b)
    return float(np.max(np.abs(a - b)))


def mean_abs_diff(a, b):
    a = flatten_to_numpy(a)
    b = flatten_to_numpy(b)
    return float(np.mean(np.abs(a - b)))


def allclose(a, b, atol=1e-6, rtol=1e-5):
    a = flatten_to_numpy(a)
    b = flatten_to_numpy(b)
    return bool(np.allclose(a, b, atol=atol, rtol=rtol))


def main():
    if len(sys.argv) != 4:
        print("Usage: python tests/compare_outputs.py <python_output_json> <node_output_json> <report_output_json>")
        sys.exit(1)

    python_path = sys.argv[1]
    node_path = sys.argv[2]
    report_path = sys.argv[3]

    with open(python_path, "r", encoding="utf-8") as f:
        py_results = json.load(f)

    with open(node_path, "r", encoding="utf-8") as f:
        node_results = json.load(f)

    if len(py_results) != len(node_results):
        raise ValueError("Python and Node output counts do not match.")

    comparisons = []
    overall_pass = True

    for py_item, node_item in zip(py_results, node_results):
        py_out = py_item["output"]
        node_out = node_item["output"]

        case_report = {
            "case_id": py_item["case_id"],
            "input": py_item["input"],
            "python_output": py_out,
            "node_output": node_out,
            "max_abs_diff": max_abs_diff(py_out, node_out),
            "mean_abs_diff": mean_abs_diff(py_out, node_out),
            "pass": allclose(py_out, node_out, atol=1e-6, rtol=1e-5)
        }

        if not case_report["pass"]:
            overall_pass = False

        comparisons.append(case_report)

    report = {
        "overall_pass": overall_pass,
        "total_cases": len(comparisons),
        "details": comparisons
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Compare report saved to: {report_path}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()