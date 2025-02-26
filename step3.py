# TDGC analysis, functions for notebooks need sub-tasking
# corresponding to Chris's notebook 3

import time
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd

import warnings

warnings.filterwarnings("ignore")

import NotebookAnalysisUtilities as NAU
import DataUtilities3.MPL_rella as RMPL
import DataUtilities3.RellaColor as RC

RMPL.SetTheme("EIGHT")
saver = RMPL.SaveFigure().savefig
KOL = RC.SeventiesFunk(brtValue=0.25)

import picarro_xarray.accessors.zero_reference_accessor  # noqa
from spectral_toolkit.db_clients.mongo_spectral_library import (
    get_mongo_spectral_library_client,
)
from spectral_toolkit.model_function.mongodb import model_function
from dask.diagnostics import ProgressBar
from dask import delayed, compute

# define compound hunters
import picarro_xarray.accessors.compound_search_accessor  # noqa

from constants import SPLIT_TASK


def notebook3_part(n):
    print(
        "********** start step 3 part %s: ********** ComputeAverageSpectraInBin ***" % n
    )
    t0 = time.time()

    config = NAU.load_analysis_yaml()
    dirs = NAU.AnalysisPaths(config["paths"])

    EXPT_START = config["times"]["expt_start"]
    EXPT_END = config["times"]["expt_end"]

    FITTING = config["fitting"]
    nu_min = FITTING["nu_min"]
    nu_max = FITTING["nu_max"]
    nominal_pressure = FITTING["nominal_pressure"]
    DONT_FIT = FITTING["dont_fit"]
    exclude_list = FITTING["exclude_list"]
    use_exact_pressure = FITTING["use_exact_pressure"]

    NUM_REFS_TO_AVE = config["valve_info"]["num_reference_periods"]

    chromatogram_duration = config["chromatogram_params"]["chromatogram_duration"]
    ave_width = config["average_spectra"]["step_width_sec"]
    ave_step = config["average_spectra"]["step_size_sec"]
    ave_na_thresh = config["average_spectra"]["na_thresh"]
    ave_pfi_thresh = config["average_spectra"]["pfi_threshold"]

    TIMES = np.arange(0, chromatogram_duration * 60, ave_step)

    # get model functions
    client = get_mongo_spectral_library_client()
    cids = client.EmpiricalSpectra.distinct("cid")

    @delayed
    def get_model_function(cid):
        try:
            return (
                cid,
                model_function(
                    client,
                    cid=cid,
                    nu_min=nu_min,
                    nu_max=nu_max,
                    nominal_pressure=nominal_pressure,
                    use_exact_pressure=use_exact_pressure,
                ),
            )
        except Exception as e:
            print(f"Unable to acquire model function for {cid}: {e}")

    tasks = [get_model_function(cid) for cid in cids]

    with ProgressBar():
        results = compute(*tasks)

    client.client.close()
    print("len(results)", len(results))

    model_functions = {}
    for cid, result in results:
        if result is not None and cid not in DONT_FIT:
            model_functions[cid] = result

    # Get data from zarr store
    ds = xr.open_zarr(dirs.zarr_path)
    ds.close()
    ds = ds.sel(_datetime=slice(EXPT_START, EXPT_END))
    # load chromatogram transitions
    df_time = pd.read_parquet(dirs.misc_results_folder / "chromatogram_times.parquet")
    # print(df_time)

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

    # compute average of spectral regions
    print("total len of TIMES", len(TIMES))
    pfi = zero_ref_ds.sel(fitter_var="partial_fit_integral").fitter_values.values
    KOL.resetCycle()

    # cut to 2~4 parts: every n element
    A = TIMES
    TIMES = A[n - 1 :: SPLIT_TASK]
    print("this part, len of TIMES", len(TIMES))

    for ctime in TIMES:
        center = pd.Timedelta(ctime, "sec")
        left = center - pd.Timedelta(ave_width / 2, "sec")
        right = center + pd.Timedelta(ave_width / 2, "sec")

        # select data in this time range
        data_mask = None
        metadata = []
        for index, row in df_time.iterrows():
            if not row["good"]:
                continue
            this_mask = (zero_ref_ds._datetime >= row["start"] + left) & (
                zero_ref_ds._datetime < row["start"] + right
            )
            this_pfi_mean = pfi[this_mask].mean()

            include_me = this_pfi_mean >= ave_pfi_thresh
            info = {
                "index": index,
                "start": row["start"],
                "sample_points": this_mask.values.sum(),
                "mean_pfi": this_pfi_mean,
                "included": include_me,
            }
            metadata.append(info)
            if include_me:
                if data_mask is None:
                    data_mask = this_mask
                else:
                    data_mask |= this_mask

        if data_mask is None:
            continue
        print(center.total_seconds(), pfi[data_mask].mean())

        timebin_dir = Path(
            dirs.average_spectra_folder / f"BIN_{int(center.total_seconds()):05d}_sec"
        )
        timebin_dir.mkdir(exist_ok=True, parents=True)

        meta_df = pd.DataFrame(metadata)
        meta_df.to_csv(timebin_dir / "metadata.csv")

        analysis_ds = zero_ref_ds.sel(_datetime=data_mask).compute()
        analysis_ds = analysis_ds.dropna(
            dim="mode", subset=["spectral_values"], thresh=ave_na_thresh
        )  # why are we keeping nan?

        analysis_ds.compound_search.set_model_functions(model_functions)
        X = analysis_ds.compound_search.X
        Y = analysis_ds.compound_search.Y
        bad_nu = Y.isna().sum(axis=1) == 1
        if bad_nu.sum() > 0:
            print(bad_nu.sum())
        X.to_parquet(timebin_dir / "X.parquet")
        Y.to_parquet(timebin_dir / "Y.parquet")
        A, F = RMPL.Maker()
        Ym = Y.mean(axis=1)
        A.scatter(Ym.index, Ym.values, c=KOL[0], s=15)
        RMPL.setLabels(
            A,
            RMPL.wavenumbers,
            "absorption [ppb/cm]",
            f"{timebin_dir.parent.name}: {timebin_dir.name}",
        )
        this_saver = RMPL.SaveFigure(DIR=timebin_dir).savefig
        this_saver(F, f"Fig_{timebin_dir.name}.png", dpi=150, closeMe=True)

    t = int(time.time() - t0)
    print("\n****************************************** ")
    print("**  Step  3 part %s done, took %.2f min ***" % (n, t / 60))
    return t


if __name__ == "__main__":
    notebook3_part(2)
