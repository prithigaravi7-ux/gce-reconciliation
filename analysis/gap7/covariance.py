
import pandas as pd
from pathlib import Path

# Locate the repository root
ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "outputs/gap7/covariance_input.csv"


def read_results(file_path):
    data = pd.read_csv(file_path)
    print("Original rows:", len(data))
    return data


def filter_results(data):
    data = data.copy()

    # Keep only admissible scientific results
    admissible = (
        data["admissible"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("true")
    )

    valid_status = (
        data["status"]
        .astype(str)
        .str.startswith("OK")
    )

    valid_mask = ~data["mask"].isin(["A", "C"])

    data = data[
        admissible & valid_status & valid_mask
    ].copy()

    print("Rows after validity filters:", len(data))

    # Gap 5 Pohl22 reproduces Gap 1 Level 2.
    # Temporarily exclude the overlapping Gap 5 family.
    duplicate_gap5 = (
        (data["gap"] == 5) &
        (data["iem_family"] == 2)
    )

    print("Gap 5 Pohl22 rows excluded:",
          duplicate_gap5.sum())

    data = data[~duplicate_gap5].copy()

    return data


def summarize_results(data):
    statistic = "dAIC_vs_best_gnfw"

    summary = (
        data.groupby(
            ["template_class", "template"]
        )[statistic]
        .agg(["mean", "var", "count"])
        .reset_index()
    )

    return summary


def main():
    data = read_results(INPUT_FILE)

    data = filter_results(data)

    print("\nRemaining rows:", len(data))
    print("\nRows by gap and IEM family:")
    print(pd.crosstab(
        data["gap"], data["iem_family"]
    ))

    print("\nUnique levels per axis:")
    for axis in [
        "mask", "iem_family",
        "level", "energy_bin"
    ]:
        print(axis, data[axis].nunique())

    print("\nHalos by template class:")
    print(
        data.groupby("template_class")["halo"]
        .nunique()
    )

    summary = summarize_results(data)

    output_dir = ROOT / "outputs/gap7"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary.to_csv(
        output_dir / "covariance_summary.csv",
        index=False
    )

    print("\nSummary saved successfully!")


if __name__ == "__main__":
    main()
