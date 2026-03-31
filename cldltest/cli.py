import argparse
from cldltest.benchmark import run_benchmark


def main():
    parser = argparse.ArgumentParser(prog="cldltest")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bench_parser = subparsers.add_parser("benchmark", help="Run benchmark on selected backends")
    bench_parser.add_argument("--model", required=True, help="Path to ONNX model")
    bench_parser.add_argument("--input", required=True, help="Path to input JSON")
    bench_parser.add_argument("--outdir", default="outputs", help="Directory to save outputs")
    bench_parser.add_argument(
        "--backends",
        nargs="+",
        choices=["py", "js", "java"],
        default=["py", "js", "java"],
        help="Backends to run: py js java"
    )

    args = parser.parse_args()

    if args.command == "benchmark":
        result = run_benchmark(
            model_path=args.model,
            input_json_path=args.input,
            outdir=args.outdir,
            backends=args.backends
        )

        print("========== 运行结果 ==========")
        for k, v in result["result_files"].items():
            print(f"{k:<16}: {v}")

        print("\n========== 对比结果 ==========")
        for pair_name, report in result["compare_reports"].items():
            print(f"{pair_name:<16}: {report['pass']} {report['metrics']}")


if __name__ == "__main__":
    main()