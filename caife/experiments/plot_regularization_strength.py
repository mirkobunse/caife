import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


def main(results_path, plot_path, criterion="gcc"):
    print(f"Plotting {plot_path} from {results_path}")
    if len(os.path.dirname(plot_path)) > 0:  # ensure that the directory exists
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    criterion_name = {
        "gcc": "glob. corr.",
        "pcs": "pairw. corr.",
        "unreg_gcc": "unreg. glob. corr.",
        "unreg_pcs": "unreg. pairw. corr.",
    }[criterion]
    criterion_color = {
        "gcc": "C0",
        "pcs": "C3",
        "unreg_gcc": "C4",
        "unreg_pcs": "C5",
    }[criterion]

    # read the results
    results = pd.read_csv(results_path, index_col=0)
    sources = results["source"].unique()
    targets = results["target"].unique()

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

    # initialize the plot
    fig, axs = plt.subplots(
        nrows=len(sources),
        ncols=len(targets),
        squeeze=False,
        sharex=True,
        figsize=(len(targets)*4, len(sources)*3),
    )
    for i_source, source in enumerate(sources):
        for i_target, target in enumerate(targets):
            ax = axs[i_source, i_target]
            ax_results = results.loc[
                (results["source"] == source) & (results["target"] == target)
            ]
            l1 = ax.plot(
                ax_results["tau"],
                ax_results[criterion],
                marker="",
                color=criterion_color,
                label=f"l: {criterion_name}",
            )
            ax.set_xscale("log")
            ax.set_xlabel(r"$\tau$")
            # ax.set_ylabel(criterion_name)
            ax.set_title(f"{source} → {target}")
            if criterion == "gcc":
                ax.set_ylim(bottom=0.725, top=1.025)
            elif criterion == "pcs":
                ax.set_ylim(bottom=0.075, top=0.375)
            ax2 = ax.twinx()
            if target != "real":
                l2 = ax2.plot(
                    ax_results["tau"],
                    ax_results["emd"],
                    marker="",
                    color="C1",
                    label="r: EMD",
                )
                # ax2.set_ylabel("EMD")
                ax2.set_yscale("log")
                ax2.set_ylim(bottom=5e-2, top=2e1)
            else:
                l2 = ax2.plot(
                    ax_results["tau"],
                    ax_results["ndf"],
                    marker="",
                    color="C2",
                    label="r: n_df",
                )
                # ax2.set_ylabel("n_df")
                ax2.set_ylim(bottom=1.5, top=10.5)
            ax2.grid(True, ls="--")
            best_tau = ax_results[  # minimum value of criterion for tau <= 1e0
                ax_results[criterion] == ax_results[criterion][ax_results["tau"]<=1].min()
            ]["tau"].iloc[0]
            l3 = ax.axvline(
                x=best_tau,
                color="gray",
                label=r"$\tau" + f"={best_tau:.1e}$",
            )
            lines = l1 + l2 + [l3]
            ax.legend(lines, [l.get_label() for l in lines], loc="upper right")

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
    parser.add_argument(
        "--criterion",
        type=str,
        default="gcc",
        help="correlation criterion to plot",
    )
    args = parser.parse_args()
    main(
        args.results_path,
        args.plot_path,
        args.criterion,
    )
