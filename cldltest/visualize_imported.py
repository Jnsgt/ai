import json
import os
import sys
import matplotlib.pyplot as plt


def main():
    if len(sys.argv) == 3:
        report_path = sys.argv[1]
        fig_path = sys.argv[2]
    else:
        project_root = os.path.dirname(os.path.abspath(__file__))
        report_path = os.path.join(project_root, "reports", "compare_report.json")
        fig_path = os.path.join(project_root, "reports", "max_abs_diff.png")

    if not os.path.exists(report_path):
        print(f"compare report not found: {report_path}")
        return

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    details = report.get("details", [])
    if not details:
        print("No details found in compare report.")
        return

    case_ids = [item["case_id"] for item in details]
    diffs = [item["max_abs_diff"] for item in details]

    plt.figure(figsize=(6, 4))
    plt.bar(case_ids, diffs)

    plt.xlabel("Case ID")
    plt.ylabel("Max Abs Diff")
    plt.title("Cross-Language Output Difference")

    max_diff = max(diffs) if diffs else 0.0

    if max_diff == 0.0:
        plt.ylim(0, 1)
        plt.text(
            0.5, 0.55,
            "All differences are 0.0",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
            fontsize=13
        )
        plt.text(
            0.5, 0.42,
            "Outputs are fully consistent",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
            fontsize=11
        )
    else:
        plt.ylim(0, max_diff * 1.2)

    plt.tight_layout(pad=0.8)
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Visualization saved to: {fig_path}")


if __name__ == "__main__":
    main()