import json
import numpy as np

from cldltest.utils.metrics import calc_metrics, max_abs_diff, mean_abs_diff, allclose


def compare_result_files(file_a: str, file_b: str, threshold: float = 1e-5):
    with open(file_a, "r", encoding="utf-8") as f:
        a_json = json.load(f)

    with open(file_b, "r", encoding="utf-8") as f:
        b_json = json.load(f)

    a = np.array(a_json["output_data"], dtype=np.float32)
    b = np.array(b_json["output_data"], dtype=np.float32)

    metrics = calc_metrics(a, b)

    report = {
        "runner_a": a_json["runner"],
        "runner_b": b_json["runner"],
        "shape_equal": a_json["output_shape"] == b_json["output_shape"],
        "dtype_equal": a_json["dtype"] == b_json["dtype"],
        "metrics": metrics,
        "pass": metrics["max_abs_diff"] < threshold
    }
    return report


def compare_case_lists(py_results, other_results, atol: float = 1e-6, rtol: float = 1e-5):
    """
    批量比较两组 case 结果。
    适用于 compare_outputs.py 那种“列表式输出”。
    """

    if len(py_results) != len(other_results):
        raise ValueError("两组结果的 case 数量不一致。")

    comparisons = []
    overall_pass = True

    for py_item, other_item in zip(py_results, other_results):
        py_out = py_item["output"]
        other_out = other_item["output"]

        case_metrics = {
            "max_abs_diff": max_abs_diff(py_out, other_out),
            "mean_abs_diff": mean_abs_diff(py_out, other_out),
            "pass": allclose(py_out, other_out, atol=atol, rtol=rtol)
        }

        case_report = {
            "case_id": py_item.get("case_id"),
            "input": py_item.get("input"),
            "output_a": py_out,
            "output_b": other_out,
            "metrics": case_metrics,
            "pass": case_metrics["pass"]
        }

        if not case_report["pass"]:
            overall_pass = False

        comparisons.append(case_report)

    return {
        "overall_pass": overall_pass,
        "total_cases": len(comparisons),
        "details": comparisons
    }


def compare_case_list_files(file_a: str, file_b: str, report_path: str = None,
                            atol: float = 1e-6, rtol: float = 1e-5):
    """
    读取两个“多 case 输出 JSON”文件并进行批量比较。
    """

    with open(file_a, "r", encoding="utf-8") as f:
        results_a = json.load(f)

    with open(file_b, "r", encoding="utf-8") as f:
        results_b = json.load(f)

    report = compare_case_lists(results_a, results_b, atol=atol, rtol=rtol)

    if report_path:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    return report