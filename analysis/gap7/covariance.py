import numpy as np
import pandas as pd
from pathlib import Path

def read_results(file_path):
    data = pd.read_csv(file_path, sep=r"\s+")
    return data

def compute_covariance(data):
    # Exclude withdrawn Gap 1 results
    valid_data = data[
        data["status"] != "WITHDRAWN_see_section_6.0_rigid_IEM"
    ].copy()

    # Fit-derived columns
    columns = [
        "TS",
        "lnL",
        "dAIC_vs_vvv",
        "dAIC_vs_best_gnfw",
        "signal_photons"
    ]

    # Check for enough valid results
    if len(valid_data) < 2:
        print("No valid Gap 1 results available for covariance.")
        return None

    # Convert fit-derived columns to numeric values
    numerical_data = valid_data[columns].apply(
        pd.to_numeric, errors="coerce"
    ).dropna()

    if len(numerical_data) < 2:
        print("Not enough valid numerical data for covariance.")
        return None

    covariance_matrix = np.cov(
        numerical_data.to_numpy(),
        rowvar=False
    )

    return covariance_matrix

def main():
    file_path = Path("results.txt")

    data = read_results(file_path)

    # Preserve all columns, including status and diffuse_split
    covariance_matrix = compute_covariance(data)

    if covariance_matrix is not None:
        print("Covariance Matrix:")
        print(covariance_matrix)

if __name__ == "__main__":
    main()
