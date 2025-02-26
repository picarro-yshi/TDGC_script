# TDGC analysis, functions for notebooks need sub-tasking
# corresponding to Chris's notebook 5A

from pathlib import Path
import numpy as np
import pandas as pd
import time
import copy

import warnings

warnings.filterwarnings("ignore")

import NotebookAnalysisUtilities as NAU
from SpectroscopyTools.HitranDBModule import PubChemEntry
from spectral_toolkit.plotting.spectral_hunting_plot import fit_result_plot
from IPython.display import display

# define compound hunters
from compound_hunting.transformers.r3_transformer import R3Transformer
from compound_hunting.assessment.regression_summary import regression_summary

from constants import SPLIT_TASK


def perform_solve(solver, X, Y):
    results = solver.fit(X, Y).get_feature_names_out()
    results = [int(r) for r in results]
    Xt = solver.transform(X)

    if len(results) > 0:
        coefs, scores = regression_summary(
            Xt, Y, multioutput_coefs="raw_values", multioutput_scores="raw_values"
        )
        coefs_mean = coefs.mean(axis=0)
        scores = scores.describe().T
        coefs = coefs.describe().T
        Xt.loc[:, 0] = np.ones(Xt.shape[0])
        spectral_contribution = Xt * coefs_mean

        Q_df = pd.DataFrame(
            solver.feature_importances_[solver._get_support_mask()],
            index=solver.feature_names_in_[solver._get_support_mask()],
            columns=["Q"],
        )

    else:
        # ToDo: updated for SFS
        Q_df = pd.DataFrame(columns=["Q"])
        coefs = None
        scores = None
        spectral_contribution = None
        coefs_mean = None

    return results, coefs, scores, Q_df, spectral_contribution, coefs_mean


def slice_it_up(df, exclude_list):
    data_mask = df.index > 0
    for exclude in exclude_list:
        data_mask &= (df.index <= exclude[0]) | (df.index >= exclude[1])
    return df[data_mask].copy()


def _process_peak(
    exp_subfolder, solver_name, solver, overwrite=True, dont_fit=[], exclude_list=[]
):
    if not exp_subfolder.exists():
        raise ValueError("No data found")

    X = pd.read_parquet(exp_subfolder / "X.parquet")
    X.columns = X.columns.astype(int)
    column_list = X.columns.to_list()
    for ignore_cid in dont_fit:
        if ignore_cid in column_list:
            X.drop(columns=[ignore_cid], inplace=True)
    Y = pd.read_parquet(exp_subfolder / "Y.parquet")

    X = slice_it_up(X, exclude_list)
    Y = slice_it_up(Y, exclude_list)
    Y = Y.dropna(axis=0)
    xmask = X.index < 0
    for nu in Y.index:
        xmask |= X.index == nu
    X = X[xmask].copy()

    results_folder = exp_subfolder / "results"
    if results_folder.exists() and not overwrite:
        print("Analysis already performed.")
        return None, None, None
    results_folder.mkdir(exist_ok=True, parents=True)

    results_txt_file = results_folder / f"{solver_name}.txt"

    results, coefs, scores, Q_df, spectral_contribution, coefs_mean = perform_solve(
        solver, X, Y
    )

    if len(results) == 0:
        return None, None, None

    Q_df.index.rename("CID", inplace=True)
    Q_df.sort_values("Q", ascending=False, inplace=True)
    Q_df["compound"] = [
        str(PubChemEntry(int(cid))).split(": ")[-1] for cid in Q_df.index
    ]
    Q_df.to_csv(results_folder / f"{solver_name}_Q_values.csv")

    fig = fit_result_plot(
        experimental_datapoints=Y.mean(axis=1),
        absorption_df=spectral_contribution,
        concentrations=coefs_mean,
        species_colormap=None,
        annotations=scores.loc[:, "mean"]
        .map(lambda x: format(x, "0.4g"))
        .to_string()
        .replace("\n", "<br>"),
        # title_text=f"{exp_subfolder.name} {solver_name}, {exp_subfolder.name}",
        title_text=f"{exp_subfolder} {solver_name}, {exp_subfolder}",
    )

    # write html and png
    fig_path = results_folder / f"{solver_name}.html"
    fig.write_html(fig_path)
    fig_path = results_folder / f"{solver_name}.png"
    fig.write_image(fig_path, width=1920, height=1080)

    with open(results_txt_file, "w") as file:
        file.write(f"{results}\n")
        if len(results) > 0:
            file.write("\nConcentrations\n")
            file.write(coefs.to_string())
            file.write("\n\nScores\n")
            file.write(scores.to_string())

    return coefs, scores, Q_df


def notebook5a_part(n):
    print("********* start step 5A part %s: ********** FittingAverageSpectra ***" % n)
    t0 = time.time()

    config = NAU.load_analysis_yaml()
    dirs = NAU.AnalysisPaths(config["paths"])

    FITTING = config["fitting"]
    DONT_FIT = FITTING["dont_fit"]
    exclude_list = FITTING["exclude_list"]
    always_include = FITTING["always_include"]
    allow_neg = config["average_spectra"]["allow_negatives_in_hunt"]

    # ***** NEGATIVE ONLY FIT WITH SEVERAL COMPOUNDS EXCLUDED FROM FIT
    FIT_FOLDER = dirs.average_spectra_folder
    OVERWRITE = True
    name = "Negs" if allow_neg else "PosOnly"

    subfolders = [p for p in Path(FIT_FOLDER).iterdir() if p.is_dir()]
    subfolders.sort(key=lambda s: s.name)

    # cut to 2~4 parts: every n element
    A = subfolders
    subfolders = A[n - 1 :: SPLIT_TASK]
    print("this part, len of TIMES", len(subfolders))

    for subfolder in subfolders:
        if "Fig" not in subfolder.name:
            # if subfolder.name < 'BIN_00000_sec': continue
            print(f"{subfolder.name}")
            start = time.time()

            # name_num = (subfolder.name).split('_')[1]
            # print(name_num)
            # if int(name_num) > 0:  # skip files already analyzed
            try:
                first_solver = R3Transformer(
                    Q_threshold=2,
                    dont_fit=DONT_FIT,
                    ban_negatives=not (allow_neg),
                    max_cycles=5,
                    initial_seed_cids=always_include,
                )
                print("  Solving %s" % name)
                print("   First Fit")
                coefs, scores, Q_df = _process_peak(
                    subfolder,
                    f"R3_presolve_{name}",
                    first_solver,
                    overwrite=OVERWRITE,
                    dont_fit=DONT_FIT,
                    exclude_list=exclude_list,
                )
                if Q_df is not None:
                    display(Q_df)

                    reqd_cids = list(Q_df.index)
                    reqd_cids = [int(c) for c in reqd_cids]
                else:
                    reqd_cids = None

                if reqd_cids is None:
                    reqd_cids = copy.deepcopy(always_include)
                else:
                    for always in always_include:
                        if always not in reqd_cids:
                            reqd_cids.append(always)
                print(reqd_cids)
                print()
                print("   Final Fit")
                final_solver = R3Transformer(
                    Q_threshold=0,
                    dont_fit=DONT_FIT,
                    ban_negatives=not (allow_neg),
                    initial_seed_cids=reqd_cids,
                )
                coefs, scores, Q_df = _process_peak(
                    subfolder,
                    f"R3_{name}",
                    final_solver,
                    overwrite=OVERWRITE,
                    dont_fit=DONT_FIT,
                    exclude_list=exclude_list,
                )
                if Q_df is not None:
                    display(Q_df)
            except Exception as e:
                print(e)
                print("******* Subfolder %s FAILED" % subfolder)
            print("Duration = %.0f secs" % (time.time() - start))

    t = int(time.time() - t0)
    print("\n****************************************** ")
    print("**  Step 5A part %s done, took %.2f min ***" % (n, t / 60))
    return t


if __name__ == "__main__":
    notebook5a_part(2)
