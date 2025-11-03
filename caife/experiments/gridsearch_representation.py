import argparse
import caife
import itertools
import numpy as np
import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


def main(
        output_path,
        n_jobs = 1,
        seed = 1491,
        is_test_run = False,
    ):
    print(f"Starting a lequa experiment to produce {output_path} with seed {seed}")
    if is_test_run:
        print("WARNING: this is a test run; results are not meaningful")
    if len(os.path.dirname(output_path)) > 0: # ensure that the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.random.seed(seed)

    # load the data
    df = pd.read_pickle("~/data/re-funfolding/23111.pkl")
    df = df[ # apply analysis-specific cuts
        (df["FSSFilter_13_1"] == 1) &
        (df["scores"] > 0.99) &
        (df["depth_unc"] < 60) &
        (df["zenith_unc_no_filter_02"] < 0.1)
    ]
    X = df[["range_no_filter_02"]] # proxy: propagated distance of muons through ice
    y = np.digitize( # target: (binned) muon energy at the surface
        df["energy_stop"],
        np.geomspace(6e2, 1e4, 11),
    )
    print(f"Read MC samples with shape {X.shape}")

    # optimize the proxy binning
    base_binning = caife.TreeBinning(DecisionTreeClassifier(random_state=seed))
    param_grid = {
        "tree__max_leaf_nodes": np.arange(start=11, stop=42, step=2),
        "tree__criterion": [ "gini", "entropy" ],
    }
    if is_test_run: # minimal testing configuration
        param_grid = {
            "tree__max_leaf_nodes": np.array([11, 21, 42]),
            "tree__criterion": [ "gini" ],
        }
    binning = caife.GridSearchRepresentation(
        base_binning,
        param_grid,
        is_verbose=True,
    )
    _ = binning.fit_transform(X, y) # ignore the resulting matrix

    # collect the results in preparation of a DataFrame
    results = []
    for r in binning.results_:
        results.append({
            "dussap": r["losses"]["dussap"],
            "blobel": r["losses"]["blobel"],
            "max_leaf_nodes": r["params"]["tree__max_leaf_nodes"],
            "criterion": r["params"]["tree__criterion"],
            "n_bins": r["representation"].n_bins_, # <= max_leaf_nodes
        })

    # also evaluate the original proxy_bins
    binning = caife.GridSearchRepresentation(
        caife.UnivariateBinning(np.geomspace(2e3, 8e3, 21)),
        {},
    )
    _ = binning.fit_transform(X, y)
    for r in binning.results_:
        results.append({
            "dussap": r["losses"]["dussap"],
            "blobel": r["losses"]["blobel"],
            "max_leaf_nodes": -1,
            "criterion": "",
            "n_bins": r["representation"].n_bins_,
        })

    # store the results
    results = pd.DataFrame(results).sort_values(["criterion", "max_leaf_nodes"])
    print(results)
    results.to_csv(output_path)
    print(f"{results.shape[0]} results succesfully stored at {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_path', type=str, help='path of an output *.csv file')
    parser.add_argument('--n_jobs', type=int, default=1, metavar='N',
                        help='number of concurrent jobs or 0 for all processors (default: 1)')
    parser.add_argument('--seed', type=int, default=1491, metavar='N',
                        help='random number generator seed (default: 1491)')
    parser.add_argument("--is_test_run", action="store_true")
    args = parser.parse_args()
    main(
        args.output_path,
        args.n_jobs,
        args.seed,
        args.is_test_run,
    )
