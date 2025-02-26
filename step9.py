# TDGC analysis, functions for notebooks need sub-tasking
# corresponding to Chris's notebook 9

import time
import numpy as np
import xarray as xr
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

import NotebookAnalysisUtilities as NAU
from spectral_toolkit.reprocess_combo import get_least_squares
import picarro_xarray.accessors.refit_accessor
import picarro_xarray.accessors.zero_reference_accessor

from constants import SPLIT_TASK


def notebook9_part(n):
    print("********** start step 9 part %s: ********** RefittingData ***" % n)
    t0 = time.time()

    # constants and parameters
    config = NAU.load_analysis_yaml()
    dirs = NAU.AnalysisPaths(config["paths"])

    RECIPE_PAD = config["refitting_params"]["pad_for_refitting"]
    NUM_REFS_TO_AVE = config["valve_info"]["num_reference_periods"]

    p_nominal = config["fitting"]["nominal_pressure"]
    chromatogram_duration = config["chromatogram_params"]["chromatogram_duration"]
    ave_step = config["average_spectra"]["step_size_sec"]
    HWIDTH_fit = config["refitting_params"]["refit_half_width"]

    TIMES = np.arange(0, chromatogram_duration * 60, ave_step)

    df_recipes = pd.read_parquet(
        dirs.misc_results_folder / "fit_cid_table_exclude.parquet"
    )
    mapper = {col: int(col) for col in df_recipes.columns if col != "recipe_id"}
    df_recipes.rename(columns=mapper, inplace=True)
    # print(df_recipes)

    # select an example
    X = df_recipes.loc[(slice(None), RECIPE_PAD), :][7144].dropna()
    X.droplevel(1).index.values

    ds = xr.open_zarr(dirs.zarr_path)
    ds.close()
    # print(ds)

    # load chromatogram transitions
    df_time = pd.read_parquet(dirs.misc_results_folder / "chromatogram_times.parquet")

    # perform absolute spectral zero referencing for the entire time series
    earliest_time = min([df_time["start"].min(), df_time["ref_start"].min()])
    latest_time = min([df_time["end"].max(), df_time["ref_end"].max()])

    run_ds = ds.sel(_datetime=slice(earliest_time, latest_time))  # gather all the data
    run_ds = run_ds.dropna(
        dim="mode", how="all", subset=["spectral_values"]
    )  # drop data with nans in spectral values

    # create a mask for reference data
    for index, row in df_time.iterrows():
        if not row["good"]:
            continue
        this_mask = (run_ds._datetime >= row["ref_start"]) & (
            run_ds._datetime < row["ref_end"]
        )
        if index == 0:
            zero_mask = this_mask
        else:
            zero_mask |= this_mask

    grouper = xr.where(
        zero_mask, 0, 1
    )  # define background by data before the start time
    zero_ref_ds = run_ds.zero_reference.zero_reference(
        transition_variable=grouper,
        reference_value=0,
        event_window_width=NUM_REFS_TO_AVE,
    )

    print("Using %d seconds as a pad for fitting" % RECIPE_PAD)
    these_recipes = df_recipes.loc[(slice(None), RECIPE_PAD), :]
    these_recipes = these_recipes.droplevel(["t_ave"])
    # these_recipes.drop(['recipe_id'], axis=1, inplace=True)
    # these_recipes[:30]

    N_unique_recipes = len(set(df_recipes.loc[(slice(None), RECIPE_PAD), "recipe_id"]))
    print(f"Number of unique fit recipes: {N_unique_recipes}")

    A = TIMES
    TIMES = A[n - 1 :: SPLIT_TASK]
    print("this part, len of TIMES", len(TIMES))

    for ctime in TIMES:
        center = pd.Timedelta(ctime, "sec")
        left = center - pd.Timedelta(HWIDTH_fit, "sec")
        right = center + pd.Timedelta(HWIDTH_fit, "sec")
        # print("center", center.total_seconds())

        # if ctime not in these_recipes.index: continue #no fitter for this time slot
        if ctime in these_recipes.index:
            print(ctime)

            recipe_id = these_recipes.loc[ctime, "recipe_id"]
            fit_these = these_recipes.loc[ctime].dropna().index.to_list()
            fit_these.pop(fit_these.index("recipe_id"))
            for name in ["sample", "reference"]:
                if name == "sample":
                    fn_save = dirs.refit_folder / f"{name}_lsq_{int(ctime):05d}.parquet"
                    if fn_save.exists():
                        print(
                            f"The sample at time = {int(center.total_seconds()):4d} has already been analyzed"
                        )
                        continue
                elif name == "reference":
                    fn_save = (
                        dirs.refit_folder / f"{name}_lsq_RID_{recipe_id:04d}.parquet"
                    )
                    if fn_save.exists():
                        print(
                            f"The reference for recipe_id = {recipe_id} has already been analyzed"
                        )
                        continue

                # select data in this time range
                if name == "sample":
                    for index, row in df_time.iterrows():
                        this_mask = (zero_ref_ds._datetime >= row["start"] + left) & (
                            zero_ref_ds._datetime < row["start"] + right
                        )
                        if index == 0:
                            data_mask = this_mask
                        else:
                            data_mask |= this_mask
                elif name == "reference":  # reference
                    for index, row in df_time.iterrows():
                        this_mask = (zero_ref_ds._datetime >= row["ref_start"]) & (
                            zero_ref_ds._datetime < row["ref_end"]
                        )
                        if index == 0:
                            data_mask = this_mask
                        else:
                            data_mask |= this_mask

                this_slot = zero_ref_ds.sel(_datetime=data_mask).compute()
                recipe_id = these_recipes.loc[ctime, "recipe_id"]
                fit_these = these_recipes.loc[ctime].dropna().index.to_list()
                fit_these.pop(fit_these.index("recipe_id"))

                fit_these = [int(e) for e in fit_these]
                # print('%s: refitting with cids: %s (recipe_id = %d)' % (name, fit_these, recipe_id))
                lsq = get_least_squares(fit_these, nominal_pressure=p_nominal)
                result_lsq = this_slot.refit.llsq(lsq=lsq, fast=False).compute()
                df_result = result_lsq.to_pandas()
                df_result["recipe_id"] = recipe_id
                df_result.to_parquet(fn_save)

    t = int(time.time() - t0)
    print("\n****************************************** ")
    print("**  Step  9 part %s done, took %.2f min ***" % (n, t / 60))
    return t


if __name__ == "__main__":
    notebook9_part(2)
