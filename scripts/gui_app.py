import json
import os
import subprocess
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
DEFAULT_MODEL = os.path.join(PROJECT_ROOT, "examples", "linear_model.onnx")
DEFAULT_INPUT = os.path.join(PROJECT_ROOT, "examples", "test_input.json")


class BenchmarkGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("cldltest - 跨语言 ONNX 对比工具")
        self.root.geometry("1180x800")
        self.root.minsize(1020, 720)

        self.model_path = tk.StringVar(value=DEFAULT_MODEL if os.path.exists(DEFAULT_MODEL) else "")
        self.input_path = tk.StringVar(value=DEFAULT_INPUT if os.path.exists(DEFAULT_INPUT) else "")
        self.outdir_path = tk.StringVar(value=OUTPUTS_DIR)
        self.threshold = tk.StringVar(value="1e-5")
        self.status_text = tk.StringVar(value="就绪")

        self.enable_python = tk.BooleanVar(value=True)
        self.enable_js = tk.BooleanVar(value=True)
        self.enable_java = tk.BooleanVar(value=True)

        self.report_labels = {}
        self.metric_labels = {}
        self.card_frames = {}

        self._build_ui()

    def _build_ui(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 20, "bold"))
        style.configure("SubTitle.TLabel", font=("Microsoft YaHei UI", 10))
        style.configure("CardTitle.TLabel", font=("Microsoft YaHei UI", 11, "bold"))
        style.configure("Pass.TLabel", foreground="#14804A", font=("Microsoft YaHei UI", 12, "bold"))
        style.configure("Fail.TLabel", foreground="#C62828", font=("Microsoft YaHei UI", 12, "bold"))
        style.configure("Idle.TLabel", foreground="#666666", font=("Microsoft YaHei UI", 12, "bold"))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        main = ttk.Frame(self.root, padding=14)
        main.grid(row=0, column=0, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(2, weight=1)

        ttk.Label(main, text="cldltest 图形化对比界面", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            main,
            text="在 Python / JavaScript / Java 三种后端中运行同一个 ONNX 模型，并自动比较结果是否一致。",
            style="SubTitle.TLabel"
        ).grid(row=1, column=0, sticky="w", pady=(2, 12))

        body = ttk.Frame(main)
        body.grid(row=2, column=0, sticky="nsew")
        body.columnconfigure(0, weight=2)
        body.columnconfigure(1, weight=3)
        body.rowconfigure(0, weight=1)

        left = ttk.LabelFrame(body, text="参数配置", padding=12)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left.columnconfigure(1, weight=1)
        left.rowconfigure(7, weight=1)

        ttk.Label(left, text="ONNX 模型文件").grid(row=0, column=0, sticky="w", pady=6)
        ttk.Entry(left, textvariable=self.model_path).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(left, text="选择", command=self.browse_model).grid(row=0, column=2, sticky="ew")

        ttk.Label(left, text="输入 JSON 文件").grid(row=1, column=0, sticky="w", pady=6)
        ttk.Entry(left, textvariable=self.input_path).grid(row=1, column=1, sticky="ew", padx=8)
        ttk.Button(left, text="选择", command=self.browse_input).grid(row=1, column=2, sticky="ew")

        ttk.Label(left, text="输出目录").grid(row=2, column=0, sticky="w", pady=6)
        ttk.Entry(left, textvariable=self.outdir_path).grid(row=2, column=1, sticky="ew", padx=8)
        ttk.Button(left, text="选择", command=self.browse_outdir).grid(row=2, column=2, sticky="ew")

        ttk.Label(left, text="误差阈值").grid(row=3, column=0, sticky="w", pady=6)
        ttk.Entry(left, textvariable=self.threshold).grid(row=3, column=1, sticky="ew", padx=8)
        ttk.Label(left, text="例如：1e-5").grid(row=3, column=2, sticky="w")

        backend_frame = ttk.LabelFrame(left, text="选择参与对比的后端", padding=10)
        backend_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(10, 8))
        ttk.Checkbutton(backend_frame, text="Python", variable=self.enable_python).grid(row=0, column=0, sticky="w", padx=(0, 12))
        ttk.Checkbutton(backend_frame, text="JavaScript", variable=self.enable_js).grid(row=0, column=1, sticky="w", padx=(0, 12))
        ttk.Checkbutton(backend_frame, text="Java", variable=self.enable_java).grid(row=0, column=2, sticky="w")

        action_frame = ttk.Frame(left)
        action_frame.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(10, 10))
        for i in range(4):
            action_frame.columnconfigure(i, weight=1)

        self.run_btn = ttk.Button(action_frame, text="开始对比", command=self.run_benchmark_thread)
        self.run_btn.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(action_frame, text="打开输出目录", command=self.open_outputs).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(action_frame, text="刷新结果", command=self.refresh_results).grid(row=0, column=2, sticky="ew", padx=6)
        ttk.Button(action_frame, text="清空日志", command=self.clear_log).grid(row=0, column=3, sticky="ew", padx=(6, 0))

        status_frame = ttk.LabelFrame(left, text="当前状态", padding=12)
        status_frame.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(4, 10))
        ttk.Label(status_frame, textvariable=self.status_text, font=("Microsoft YaHei UI", 10, "bold")).grid(row=0, column=0, sticky="w")

        file_frame = ttk.LabelFrame(left, text="当前输出文件说明", padding=12)
        file_frame.grid(row=7, column=0, columnspan=3, sticky="nsew")
        file_frame.columnconfigure(1, weight=1)

        self._add_file_row(file_frame, 0, "Python 结果", "python_result.json")
        self._add_file_row(file_frame, 1, "JavaScript 结果", "js_result.json")
        self._add_file_row(file_frame, 2, "Java 结果", "java_result.json")
        self._add_file_row(file_frame, 3, "Py vs JS", "compare_py_js.json")
        self._add_file_row(file_frame, 4, "Py vs Java", "compare_py_java.json")
        self._add_file_row(file_frame, 5, "JS vs Java", "compare_js_java.json")

        right = ttk.Frame(body)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        right.rowconfigure(2, weight=2)

        summary = ttk.LabelFrame(right, text="对比结果卡片", padding=12)
        summary.grid(row=0, column=0, sticky="ew")
        for i in range(3):
            summary.columnconfigure(i, weight=1)

        self._build_card(summary, 0, "Py vs JS")
        self._build_card(summary, 1, "Py vs Java")
        self._build_card(summary, 2, "JS vs Java")

        table_frame = ttk.LabelFrame(right, text="结果表格", padding=12)
        table_frame.grid(row=1, column=0, sticky="nsew", pady=(10, 10))
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        columns = ("pair", "status", "max_abs", "mean_abs", "max_rel")
        self.result_table = ttk.Treeview(table_frame, columns=columns, show="headings", height=8)
        self.result_table.heading("pair", text="对比组")
        self.result_table.heading("status", text="状态")
        self.result_table.heading("max_abs", text="最大绝对误差")
        self.result_table.heading("mean_abs", text="平均绝对误差")
        self.result_table.heading("max_rel", text="最大相对误差")

        self.result_table.column("pair", width=120, anchor="center")
        self.result_table.column("status", width=90, anchor="center")
        self.result_table.column("max_abs", width=150, anchor="center")
        self.result_table.column("mean_abs", width=150, anchor="center")
        self.result_table.column("max_rel", width=150, anchor="center")

        self.result_table.grid(row=0, column=0, sticky="nsew")
        table_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.result_table.yview)
        table_scroll.grid(row=0, column=1, sticky="ns")
        self.result_table.configure(yscrollcommand=table_scroll.set)

        bottom = ttk.Panedwindow(right, orient=tk.HORIZONTAL)
        bottom.grid(row=2, column=0, sticky="nsew")

        preview_frame = ttk.LabelFrame(bottom, text="结果文件预览", padding=10)
        log_frame = ttk.LabelFrame(bottom, text="运行日志", padding=10)
        bottom.add(preview_frame, weight=1)
        bottom.add(log_frame, weight=1)

        preview_frame.rowconfigure(0, weight=1)
        preview_frame.columnconfigure(0, weight=1)
        self.preview = tk.Text(preview_frame, wrap="word")
        self.preview.grid(row=0, column=0, sticky="nsew")
        preview_scroll = ttk.Scrollbar(preview_frame, orient="vertical", command=self.preview.yview)
        preview_scroll.grid(row=0, column=1, sticky="ns")
        self.preview.configure(yscrollcommand=preview_scroll.set)

        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.log_box = tk.Text(log_frame, wrap="word")
        self.log_box.grid(row=0, column=0, sticky="nsew")
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_box.yview)
        log_scroll.grid(row=0, column=1, sticky="ns")
        self.log_box.configure(yscrollcommand=log_scroll.set)

        self.refresh_results()

    def _add_file_row(self, parent, row, title, filename):
        ttk.Label(parent, text=title + "：").grid(row=row, column=0, sticky="w", pady=2)
        ttk.Label(parent, text=filename).grid(row=row, column=1, sticky="w", pady=2)

    def _build_card(self, parent, column, pair_name):
        frame = tk.Frame(parent, bg="#F6F8FA", bd=1, relief="solid")
        frame.grid(row=0, column=column, sticky="nsew", padx=6)
        frame.grid_columnconfigure(0, weight=1)

        title = ttk.Label(frame, text=pair_name, style="CardTitle.TLabel")
        title.grid(row=0, column=0, sticky="w", padx=12, pady=(10, 6))

        status = ttk.Label(frame, text="未运行", style="Idle.TLabel")
        status.grid(row=1, column=0, sticky="w", padx=12)

        metrics = ttk.Label(frame, text="max_abs_diff=-\nmean_abs_diff=-\nmax_rel_diff=-", justify="left")
        metrics.grid(row=2, column=0, sticky="w", padx=12, pady=(8, 12))

        self.card_frames[pair_name] = frame
        self.report_labels[pair_name] = status
        self.metric_labels[pair_name] = metrics

    def log(self, text: str):
        self.log_box.insert("end", text + "\n")
        self.log_box.see("end")
        self.root.update_idletasks()

    def clear_log(self):
        self.log_box.delete("1.0", "end")

    def browse_model(self):
        path = filedialog.askopenfilename(
            title="选择 ONNX 模型文件",
            filetypes=[("ONNX 文件", "*.onnx"), ("所有文件", "*.*")]
        )
        if path:
            self.model_path.set(path)

    def browse_input(self):
        path = filedialog.askopenfilename(
            title="选择输入 JSON 文件",
            filetypes=[("JSON 文件", "*.json"), ("所有文件", "*.*")]
        )
        if path:
            self.input_path.set(path)

    def browse_outdir(self):
        path = filedialog.askdirectory(title="选择输出目录")
        if path:
            self.outdir_path.set(path)

    def open_outputs(self):
        outdir = self.outdir_path.get().strip() or OUTPUTS_DIR
        outdir = os.path.abspath(outdir)
        os.makedirs(outdir, exist_ok=True)
        os.startfile(outdir)

    def validate_inputs(self):
        model = self.model_path.get().strip()
        input_json = self.input_path.get().strip()
        outdir = self.outdir_path.get().strip()

        if not model or not os.path.isfile(model):
            messagebox.showerror("模型文件无效", "请选择一个有效的 ONNX 模型文件。")
            return None
        if not input_json or not os.path.isfile(input_json):
            messagebox.showerror("输入文件无效", "请选择一个有效的输入 JSON 文件。")
            return None
        if not outdir:
            messagebox.showerror("输出目录无效", "请选择一个输出目录。")
            return None
        if not (self.enable_python.get() or self.enable_js.get() or self.enable_java.get()):
            messagebox.showerror("未选择后端", "请至少勾选一个需要运行的后端。")
            return None
        try:
            float(self.threshold.get().strip())
        except ValueError:
            messagebox.showerror("误差阈值无效", "误差阈值必须是合法数字，例如 1e-5。")
            return None

        os.makedirs(outdir, exist_ok=True)
        return model, input_json, outdir

    def run_benchmark_thread(self):
        validated = self.validate_inputs()
        if not validated:
            return

        self.run_btn.config(state="disabled")
        self.status_text.set("正在运行对比任务...")
        self.log("=" * 72)
        self.log("开始执行对比任务...")
        t = threading.Thread(target=self._run_benchmark_worker, args=validated, daemon=True)
        t.start()

    def _run_benchmark_worker(self, model, input_json, outdir):
        try:
            cmd = [
                "python", "-m", "cldltest.cli", "benchmark",
                "--model", model,
                "--input", input_json,
                "--outdir", outdir,
            ]
            selected_backends = []
            if self.enable_python.get():
                selected_backends.append("py")
            if self.enable_js.get():
                selected_backends.append("js")
            if self.enable_java.get():
                selected_backends.append("java")

            cmd.append("--backends")
            cmd.extend(selected_backends)
            

            self.log("执行命令：" + " ".join(cmd))
            proc = subprocess.run(
                cmd,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace"
            )

            if proc.stdout:
                self.log("【标准输出】")
                self.log(proc.stdout.strip())
            if proc.stderr:
                self.log("【错误输出】")
                self.log(proc.stderr.strip())

            if proc.returncode != 0:
                self.root.after(0, lambda: self._finish_run(False, f"对比任务失败（返回码 {proc.returncode}）"))
                return

            self.root.after(0, self.refresh_results)
            self.root.after(0, lambda: self._finish_run(True, "对比任务执行完成"))
        except Exception as e:
            self.root.after(0, lambda: self._finish_run(False, f"错误：{e}"))

    def _finish_run(self, success: bool, message: str):
        self.status_text.set(message)
        self.run_btn.config(state="normal")
        if success:
            messagebox.showinfo("完成", message)
        else:
            messagebox.showerror("错误", message)

    def _set_card_status(self, pair, status_text, passed=None):
        lbl = self.report_labels[pair]
        if passed is True:
            lbl.configure(text=status_text, style="Pass.TLabel")
        elif passed is False:
            lbl.configure(text=status_text, style="Fail.TLabel")
        else:
            lbl.configure(text=status_text, style="Idle.TLabel")

    def refresh_results(self):
        outdir = os.path.abspath(self.outdir_path.get().strip() or OUTPUTS_DIR)
        os.makedirs(outdir, exist_ok=True)

        enabled_map = {
            "Py vs JS": self.enable_python.get() and self.enable_js.get(),
            "Py vs Java": self.enable_python.get() and self.enable_java.get(),
            "JS vs Java": self.enable_js.get() and self.enable_java.get(),
        }

        compare_files = {
            "Py vs JS": os.path.join(outdir, "compare_py_js.json"),
            "Py vs Java": os.path.join(outdir, "compare_py_java.json"),
            "JS vs Java": os.path.join(outdir, "compare_js_java.json"),
        }

        for item in self.result_table.get_children():
            self.result_table.delete(item)

        for pair, path in compare_files.items():
            if not enabled_map[pair]:
                self._set_card_status(pair, "未启用", None)
                self.metric_labels[pair].configure(text="该对比组未被当前勾选的后端组合启用。")
                self.result_table.insert("", "end", values=(pair, "未启用", "-", "-", "-"))
                continue

            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    passed = data.get("pass", False)
                    metrics = data.get("metrics", {})
                    self._set_card_status(pair, "通过" if passed else "失败", passed)
                    self.metric_labels[pair].configure(
                        text=(
                            f"max_abs_diff={metrics.get('max_abs_diff', '-')}\n"
                            f"mean_abs_diff={metrics.get('mean_abs_diff', '-')}\n"
                            f"max_rel_diff={metrics.get('max_rel_diff', '-')}"
                        )
                    )
                    self.result_table.insert(
                        "", "end",
                        values=(
                            pair,
                            "通过" if passed else "失败",
                            metrics.get("max_abs_diff", "-"),
                            metrics.get("mean_abs_diff", "-"),
                            metrics.get("max_rel_diff", "-")
                        )
                    )
                except Exception as e:
                    self._set_card_status(pair, "错误", False)
                    self.metric_labels[pair].configure(text=str(e))
                    self.result_table.insert("", "end", values=(pair, "错误", "-", "-", "-"))
            else:
                self._set_card_status(pair, "未运行", None)
                self.metric_labels[pair].configure(text="尚未生成该对比结果文件。")
                self.result_table.insert("", "end", values=(pair, "未运行", "-", "-", "-"))

        preview_candidates = []
        if self.enable_python.get():
            preview_candidates.append(os.path.join(outdir, "python_result.json"))
        if self.enable_js.get():
            preview_candidates.append(os.path.join(outdir, "js_result.json"))
        if self.enable_java.get():
            preview_candidates.append(os.path.join(outdir, "java_result.json"))
        for p in compare_files.values():
            preview_candidates.append(p)

        preview_text = "暂未发现结果文件，请先运行一次对比任务。"
        for p in preview_candidates:
            if os.path.exists(p):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        preview_text = f"当前预览：{os.path.basename(p)}\n\n" + f.read()
                    break
                except Exception as e:
                    preview_text = f"读取预览文件失败：{e}"
                    break

        self.preview.delete("1.0", "end")
        self.preview.insert("1.0", preview_text)


def main():
    root = tk.Tk()
    BenchmarkGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
