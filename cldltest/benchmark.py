import json
import os
import shutil
import subprocess
from itertools import combinations

from cldltest.runners.python_runner import run_python_backend
from cldltest.comparator import compare_result_files


BACKEND_NAME_MAP = {
    "py": "python",
    "js": "js",
    "java": "java",
}

COMPARE_FILE_MAP = {
    ("py", "js"): "compare_py_js.json",
    ("py", "java"): "compare_py_java.json",
    ("js", "java"): "compare_js_java.json",
}


def run_benchmark(model_path: str, input_json_path: str, outdir: str, backends=None):
    if backends is None:
        backends = ["py", "js", "java"]

    backends = list(dict.fromkeys(backends))  # 去重，保序

    os.makedirs(outdir, exist_ok=True)

    model_path = os.path.abspath(model_path)
    input_json_path = os.path.abspath(input_json_path)
    outdir = os.path.abspath(outdir)

    result_files = {}
    compare_reports = {}

    # 1. 跑 Python backend
    if "py" in backends:
        py_out = os.path.join(outdir, "python_result.json")
        run_python_backend(model_path, input_json_path, py_out)
        result_files["python_result"] = py_out

    # 2. 跑 JS backend
    if "js" in backends:
        js_out = os.path.join(outdir, "js_result.json")
        js_runner_path = os.path.join(os.path.dirname(__file__), "runners", "js_runner.js")
        js_runner_path = os.path.abspath(js_runner_path)
        cmd_js = ["node", js_runner_path, model_path, input_json_path, js_out]
        subprocess.run(cmd_js, check=True)
        result_files["js_result"] = js_out

    # 3. 跑 Java backend
    if "java" in backends:
        java_out = os.path.join(outdir, "java_result.json")
        java_runner_dir = os.path.join(os.path.dirname(__file__), "runners", "java_runner")
        java_runner_dir = os.path.abspath(java_runner_dir)

        if not os.path.isdir(java_runner_dir):
            raise FileNotFoundError(f"Java runner 目录不存在: {java_runner_dir}")

        mvn_exe = shutil.which("mvn.cmd") or shutil.which("mvn")
        if mvn_exe is None:
            raise FileNotFoundError("找不到 Maven 命令，请确认 mvn 或 mvn.cmd 已加入 PATH")

        cmd_java = [
            mvn_exe,
            "-U",
            "exec:java",
            '-Dexec.mainClass=OnnxJavaRunner',
            f'-Dexec.args={model_path} {input_json_path} {java_out}'
        ]
        subprocess.run(cmd_java, check=True, cwd=java_runner_dir)
        result_files["java_result"] = java_out

    # 4. 两两比较（只比较已运行的后端）
    for a, b in combinations(backends, 2):
        pair = tuple(sorted((a, b), key=lambda x: ["py", "js", "java"].index(x)))

        if pair not in COMPARE_FILE_MAP:
            continue

        file_a = None
        file_b = None

        if a == "py":
            file_a = result_files.get("python_result")
        elif a == "js":
            file_a = result_files.get("js_result")
        elif a == "java":
            file_a = result_files.get("java_result")

        if b == "py":
            file_b = result_files.get("python_result")
        elif b == "js":
            file_b = result_files.get("js_result")
        elif b == "java":
            file_b = result_files.get("java_result")

        if not file_a or not file_b:
            continue

        report = compare_result_files(file_a, file_b)

        compare_filename = COMPARE_FILE_MAP[pair]
        compare_path = os.path.join(outdir, compare_filename)

        with open(compare_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        compare_reports[f"{pair[0]}_vs_{pair[1]}"] = report

    return {
        "result_files": result_files,
        "compare_reports": compare_reports,
    }