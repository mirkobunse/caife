import argparse
import matplotlib.pyplot as plt
import os
import pandas as pd


def main(results_path, plot_path):
    print(f"Plotting {plot_path} from {results_path}")
    if len(os.path.dirname(plot_path)) > 0:  # ensure that the directory exists
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    # read the results
    results = pd.read_csv(results_path, index_col=0)
    sources = results["source"].unique()
    targets = results["target"].unique()

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
                ax_results["gcc"],  # = global_correlation_coefficient
                marker="",
                color="C0",
                label="global corr. (l)",
            )
            ax.set_xscale("log")
            ax.set_xlabel(r"$\tau$")
            # ax.set_ylabel("global corr.")
            ax.set_title(f"{source} → {target}")
            ax.grid(True, which="both", ls="--")
            ax2 = ax.twinx()
            if target != "real":
                l2 = ax2.plot(
                    ax_results["tau"],
                    ax_results["emd"],
                    marker="",
                    color="C1",
                    label="EMD (r)",
                )
                # ax2.set_ylabel("EMD")
                ax2.set_yscale("log")
            else:
                l2 = ax2.plot(
                    ax_results["tau"],
                    ax_results["ndf"],
                    marker="",
                    color="C2",
                    label="n_df (r)",
                )
                # ax2.set_ylabel("n_df")
            best_tau = ax_results[  # minimum gcc for tau <= 1e0
                ax_results["gcc"] == ax_results["gcc"][ax_results["tau"]<=1].min()
            ]["tau"].iloc[0]
            l3 = ax.axvline(
                x=best_tau,
                color="gray",
                label=f"tau={best_tau:.3e}",
            )
            lines = l1 + l2 + [l3]
            ax.legend(lines, [l.get_label() for l in lines])

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
