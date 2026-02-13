import argparse
import caife
import itertools
import numpy as np
import os
import pandas as pd
import time
from datetime import datetime, timedelta
from jax import numpy as jnp
from sklearn.model_selection import train_test_split
from tqdm import tqdm


PRIMARY_MODELS = [
    "GaisserH3a",
    "GaisserH4a",
    "Hoerandel5",
    "Honda2004",
    "GlobalFitGST",
]

TARGET_BINS = np.concatenate(  # target bin boundaries, including overflow bins
    ([-np.inf], np.geomspace(6e2, 1e4, 11), [np.inf])
)

PROXY_BINNING = caife.UnivariateBinning(  # proxy bin boundaries
    np.concatenate(([-np.inf], np.geomspace(2e3, 8e3, 21), [np.inf])),
)

A_EFF = np.array([  # effective area (ignoring it's std for now)
    645520.04714621,
    1212134.8205301,
    1199176.36234776,
    1162604.38910306,
    1199210.00058835,
    1276227.52332754,
    1333565.11305509,
    1386611.85910555,
    1426714.41354991,
    1512582.05114251,
])


def read_data(path, is_training_data):
    df = pd.read_pickle(path)
    df = df[  # apply analysis-specific cuts
        (df["FSSFilter_13_1"] == 1)
        & (df["scores"] > 0.99)
        & (df["depth_unc"] < 60)
        & (df["zenith_unc_no_filter"] < 0.1)
    ]
    X = df[["range_no_filter"]].to_numpy()  # proxy: propagated distance of muons
    if not is_training_data:
        livetime = np.mean(df[["livetime"]])  # constant value -> use mean
        print(f"Read {X.shape[0]} observed samples")
        return X, livetime
    y = df["energy_stop"].to_numpy()  # target: muon energy at the surface
    weights = {m: df[m].to_numpy() for m in PRIMARY_MODELS}
    S = df[[  # systematic parameters
        "Absorption",
        "DOMEfficiency",
        "Scattering",
        # "HoleIceForward_Unified_p0",
        # "HoleIceForward_Unified_p1",
    ]].to_numpy()
    i_finite = np.isfinite(y) # discard NaN targets
    X = X[i_finite]
    y = y[i_finite]
    for k in weights.keys():
        weights[k] = weights[k][i_finite]
    S = S[i_finite]
    print(f"Discarded {len(i_finite) - i_finite.sum()} NaN targets")
    print(f"Read {X.shape[0]} training samples")
    return X, y, weights, S


def main(
    output_path,
    mc_path="~/data/caife/23111_final.pkl",
    obs_path="~/data/caife/2020.pkl",
    tikhonov_scaling="none",
    seed=1491,
    is_test_run=False,
):
    print(
        "Starting a regularization_strength experiment",
        f"to produce {output_path} with seed {seed}",
    )
    if tikhonov_scaling not in ["none", "observation", "training"]:
        raise ValueError("Invalid value for tikhonov_scaling")
    if is_test_run:
        print("WARNING: this is a test run; results are not meaningful")
    if len(os.path.dirname(output_path)) > 0:  # ensure that the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.random.seed(seed)

    # load the MC training data (unzip from /cephfs_projects/caife/23111_final.pkl.zip)
    X, y, weights, S = read_data(mc_path, is_training_data=True)

    # load the real observation data (unzip from /cephfs_projects/caife/2020.pkl.zip)
    X_obs, livetime_obs = read_data(obs_path, is_training_data=False)

    # iterate over all (source, target) transfer_settings of PRIMARY_MODELS
    transfer_settings = list(itertools.product(
        PRIMARY_MODELS[:1] if is_test_run else PRIMARY_MODELS,
        [*PRIMARY_MODELS, "real"],
    ))
    sampling_state = np.random.RandomState(seed)  # RandomState for data sampling
    solver_state = np.random.RandomState(seed)  # RandomState for unfolding
    results = []  # where to store results
    for i_transfer_setting, (source, target) in enumerate(transfer_settings):
        desc = f"[{i_transfer_setting+1}/{len(transfer_settings)}]"
        t_init = time.time()

        if target != "real":
            w_source = weights[source]  # source domain weights
            w_target = weights[target]  # target domain weights

            # train-test split with a bootstrapped test set
            X_trn, X_tst, y_trn, y_tst, w_trn, _, _, w_tst, S_trn, _ = train_test_split(
                X, y, w_source, w_target, S, train_size=0.8, random_state=sampling_state
            )
            i_tst = sampling_state.choice(  # bootstrapping
                np.arange(len(y_tst)),
                p=w_tst / w_tst.sum(),
                replace=True,
                size=len(y_tst),
            )
            X_tst, y_tst = X_tst[i_tst], y_tst[i_tst]
            del w_tst  # weights should only be used for bootstrapping

        else:
            X_trn, y_trn, w_trn, S_trn = X, y, weights[source], S
            X_tst = X_obs

        # fit a caife model with systematics
        print(
            f"{desc} {datetime.now().strftime('%H:%M:%S')} |",
            f"Fitting a model for {source} → {target}...",
        )
        model = caife.LinearSystematicsCountModel(
            TARGET_BINS,
            PROXY_BINNING,
            solver_options={
                "gtol": 1e-8,
                "maxiter": 100 if is_test_run else 10_000,
            },
        )
        model.fit(X_trn, y_trn, sample_weight=w_trn, systematics=S_trn)
        print(
            f"{desc} {datetime.now().strftime('%H:%M:%S')} |",
            f"Fitting took {model.opt_.wallclock_time:.1f} s, {model.opt_.nit} it",
        )

        # define a factory for negative log-likelihood functions
        f_trn = model.target_view(y_trn, sample_weight=w_trn)  # the training spectrum
        def create_nll(model, X_tst, tau=0.0004):
            g_tst = model.proxy_view(X_tst)  # applies the binning to X
            def nll(f, s):
                g_est = model(f, s)  # consider nuisance parameters s
                value = caife.poisson_nll(g_est, g_tst)
                if tau is not None:
                    scaling_factors = None
                    if tikhonov_scaling == "observation":
                        scaling_factors = f[1:-1].min() / f[1:-1]
                    elif tikhonov_scaling == "training":
                        scaling_factors = f_trn[1:-1].min() / f_trn[1:-1]
                    reg = caife.tikhonov_regularization(
                        jnp.log(f[1:-1] / A_EFF + 1e-10),
                        scaling_factors=scaling_factors
                    )
                    value += reg / tau
                return value
            return nll
        unreg_nll = create_nll(model, X_tst, None)

        # evaluate a broad range of tau values
        tau_values = np.logspace(10, -8, 4 if is_test_run else 55)
        for tau in tqdm(tau_values, ncols=80, desc=f"{desc} Solving"):
            nll = create_nll(model, X_tst, tau)
            solver = caife.ScipySolver(
                seed=solver_state.randint(np.iinfo(np.uint32).max),
            )
            f_est = solver.solve(
                nll,
                caife.LatentSpectrum(
                    n_samples=len(X_tst), n_bins_target=model.n_bins_target
                ),
                caife.LatentSystematics(bounds=model.systematic_bounds),
            )
            if target != "real":
                f_tst = model.target_view(y_tst)  # the true solution
                emd = np.abs(np.cumsum(  # Earth Mover's Distance in log space
                    np.log10(f_tst[1:-1]) - np.log10(f_est[1:-1])
                )).sum().item()
                errors, _ = caife.uncertainty_from_hessian(f_est, nll)
                gaussian_nll = caife.gaussian_nll_score(f_tst, f_est, errors)
            else:
                emd = np.nan
                gaussian_nll = np.nan
            results.append({  # store the results
                "source": source,
                "target": target,
                "tau": tau,
                "gcc": caife.global_correlation_coefficient(f_est, nll),
                "pcs": caife.pairwise_correlation_score(f_est, nll),
                "unreg_gcc": caife.global_correlation_coefficient(f_est, unreg_nll),
                "unreg_pcs": caife.pairwise_correlation_score(f_est, unreg_nll),
                "ndf": caife.effective_number_of_degrees_of_freedom(f_est, unreg_nll, tau),
                "emd": emd,
                "gaussian_nll": gaussian_nll,
            })

        # compute ETA from the time spent in this transfer setting
        wallclock_time = time.time() - t_init
        time_remaining = (len(transfer_settings) - (i_transfer_setting+1)) * wallclock_time
        eta = datetime.now() + timedelta(seconds=time_remaining)
        print(
            f"{desc} {datetime.now().strftime('%H:%M:%S')} |",
            f"{source} → {target} took {wallclock_time:.1f} s;",
            f"ETA: {eta.strftime('%Y-%m-%d %H:%M:%S')}",
        )

    # store the results
    results = pd.DataFrame(results)
    results.to_csv(output_path)
    print(results)
    print(f"{results.shape[0]} results succesfully stored at {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output_path",
        type=str,
        help="path of an output *.csv file",
    )
    parser.add_argument(
        "--mc_path",
        type=str,
        default="~/data/caife/23111_final.pkl",
        help="path of an input *.pkl file with MC training data",
    )
    parser.add_argument(
        "--obs_path",
        type=str,
        default="~/data/caife/2020.pkl",
        help="path of an input *.pkl file with observed real data",
    )
    parser.add_argument(
        "--tikhonov_scaling",
        type=str,
        default="none",
        help="how to scale the Tikhonov matrix",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1491,
        metavar="N",
        help="random number generator seed (default: 1491)",
    )
    parser.add_argument("--is_test_run", action="store_true")
    args = parser.parse_args()
    main(
        args.output_path,
        args.mc_path,
        args.obs_path,
        args.tikhonov_scaling,
        args.seed,
        args.is_test_run,
    )
