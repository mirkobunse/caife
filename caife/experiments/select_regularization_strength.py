import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


def main(results_path, plot_path):
    print(f"Plotting {plot_path} from {results_path}")
    if len(os.path.dirname(plot_path)) > 0:  # ensure that the directory exists
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    # read the results
    results = pd.read_csv(results_path, index_col=0)
    results = results[results["target"] != "real"] # omit NaNs in EMD

    # hack: parse nasty string representations
    if results["emd"].dtype != np.float64:
        import re
        def map_bullshit_to_gold(v):
            matches = re.findall(r"\d+\.\d+", str(v))
            if str(v) == "nan":
                return np.nan
            if len(matches) > 0:
                return float(matches[0])
            print("BULLSHIT", v)
            raise
        results["emd"] = [map_bullshit_to_gold(v) for v in results["emd"]]

    # find the median tau to define some of the selection strategies
    static_strategies = ["static_min_emd", "static_min_gcc", "static_min_pcs", "static_fixed"]
    taus = {k: [] for k in static_strategies}
    for _, group in results.groupby(["source", "target"]):
        taus["static_min_emd"].append(group[  # minimum EMD
            group["emd"] == group["emd"][group["tau"]<=1].min()
        ]["tau"].iloc[0])
        taus["static_min_gcc"].append(group[  # minimum gcc for tau <= 1e0
            group["gcc"] == group["gcc"][group["tau"]<=1].min()
        ]["tau"].iloc[0])
        taus["static_min_pcs"].append(group[  # minimum pcs for tau <= 1e0
            group["pcs"] == group["pcs"][group["tau"]<=1].min()
        ]["tau"].iloc[0])
        taus["static_fixed"].append(1e-5)  # fixed to given value
    taus = {k: np.median(v) for k, v in taus.items()}

    # compute EMDs for different selection strategies
    dynamic_strategies = ["min_emd", "min_gcc", "min_pcs"]
    emds = {k: [] for k in dynamic_strategies + static_strategies}
    for _, group in results.groupby(["source", "target"]):
        emds["min_emd"].append(group[  # minimum EMD (privileged baseline)
            group["emd"] == group["emd"][group["tau"]<=1].min()
        ]["emd"].iloc[0])
        emds["min_gcc"].append(group[  # minimum gcc for tau <= 1e0
            group["gcc"] == group["gcc"][group["tau"]<=1].min()
        ]["emd"].iloc[0])
        emds["min_pcs"].append(group[  # minimum pcs for tau <= 1e0
            group["pcs"] == group["pcs"][group["tau"]<=1].min()
        ]["emd"].iloc[0])
        emds["static_min_emd"].append(group[  # static tau for minimum EMD
            group["tau"] == taus["static_min_emd"]
        ]["emd"].iloc[0])
        emds["static_min_gcc"].append(group[  # static tau for minimum gcc
            group["tau"] == taus["static_min_gcc"]
        ]["emd"].iloc[0])
        emds["static_min_pcs"].append(group[  # static tau for minimum pcs
            group["tau"] == taus["static_min_pcs"]
        ]["emd"].iloc[0])
        emds["static_fixed"].append(group[  # fixed to given value
            group["tau"] == taus["static_fixed"]
        ]["emd"].iloc[0])

    # initialize the plot
    fig, ax = plt.subplots()
    static_labels = [f"{k}\n($" + r"\tau" + f"={taus[k]:.3e}$)" for k in static_strategies]
    ax.boxplot(
        np.stack([emds[k] for k in dynamic_strategies + static_strategies]).T,
        tick_labels=dynamic_strategies + static_labels,
    )
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")
    ax.set_xlabel("selection strategy for " + r"$\tau$")
    ax.set_ylabel("EMD")
    ax.set_yscale("log")

    # export the file
    fig.tight_layout()
    fig.savefig(plot_path, transparent=True)
    print(f"Succesfully plotted {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results_path",
        type=str,
        help="path of an input *.csv file",
    )
    parser.add_argument(
        "plot_path",
        type=str,
        help="path of an output *.pdf file",
    )
    args = parser.parse_args()
    main(
        args.results_path,
        args.plot_path,
    )
