# TDGC analysis, functions for notebooks no need sub-tasking
# corresponding to Chris's notebook 1,2,6,7

import time
import numpy as np
import pandas as pd
import datetime
import pytz
import collections
from io import StringIO

import warnings

warnings.filterwarnings("ignore")

import xarray as xr
from picarro_xarray.io.read_files import pool_data_to_zarr
import NotebookAnalysisUtilities as NAU
import DataUtilities3.MPL_rella as RMPL
import DataUtilities3.RellaColor as RC

# import picarro_xarray.accessors.zero_reference_accessor

RMPL.SetTheme("EIGHT")
saver = RMPL.SaveFigure().savefig
KOL = RC.SeventiesFunk(brtValue=0.25)
from IPython.core.debugger import set_trace
from sklearn.cluster import DBSCAN

from rdkit.Chem import Draw
from rdkit import Chem
import model_function_processing.chemistry.structure as struct
from SpectroscopyTools.HitranDBModule import PubChemEntry
from model_function_processing.mongo.library_access import (
    get_mongo_spectral_library_client,
)
from model_function_processing.chemistry.display import draw_molecule_no_Hs


# ####### Step01_PoolDataAndCreateZarrStore
def notebook1():
    print("********** start step 1: ********** PoolDataAndCreateZarrStore ***")
    t0 = time.time()

    config = NAU.load_analysis_yaml("./analysis_config.yaml")
    dirs = NAU.AnalysisPaths(config["paths"])

    zarr_path = pool_data_to_zarr(
        dirs.combo_folder,
        dirs.exp_folder,
        config["main"]["time_zone"],
        None,
        config["main"]["fitter_name"],
        exist_ok=True,
    )
    # saving h5 takes long time

    ds = xr.open_zarr(zarr_path)
    ds = ds.drop_vars(["fitter_origin_file"])
    ds.close()
    print(ds)

    t = int(time.time() - t0)
    print("\n*********************************** ")
    print("**  Step  1 done, took %.2f min ***" % (t / 60))
    return t


# ####### Step02_FindChromatogramTransitions
def notebook2():
    print("********** start step 2: ********** FindChromatogramTransitions ***")
    t0 = time.time()

    # constants and parameters
    config = NAU.load_analysis_yaml()
    dirs = NAU.AnalysisPaths(config["paths"])

    plot_pad = pd.Timedelta(config["misc"]["chromatogram_plot_pad"], "min")

    chromatogram_duration = pd.Timedelta(
        config["chromatogram_params"]["chromatogram_duration"], "min"
    )
    baseline_start_trim = pd.Timedelta(
        config["chromatogram_params"]["baseline_start_trim"], "min"
    )
    baseline_end_trim = pd.Timedelta(
        config["chromatogram_params"]["baseline_end_trim"], "min"
    )
    max_data_gap = config["chromatogram_params"]["max_instrument_data_gap"]
    TDSTATE_FOR_TRIGGER = config['chromatogram_params']['baseline_state']

    NUM_REFS_TO_AVE = config["valve_info"]["num_reference_periods"]
    ref_min_duration = config["valve_info"]["ref_min_duration"]
    start_trim = pd.Timedelta(config["valve_info"]["start_trim"], "sec")
    end_trim = pd.Timedelta(config["valve_info"]["end_trim"], "sec")

    # Get data from zarr store
    ds = NAU.load_ds_from_zarr(dirs.zarr_path)
    expt_mask = (ds._datetime >= config["times"]["expt_start"]) & (
        ds._datetime <= config["times"]["expt_end"]
    )
    ds = ds.sel(_datetime=expt_mask)
    print("-> data after apply time mask")
    print(ds)

    these_items = [item for item in dirs.TDGC_logs.rglob("*.txt")]
    these_items.sort(key = lambda s: s.name)

    local_tz = pytz.timezone(config["main"]["time_zone"])
    df_gc = pd.DataFrame()
    df1 = pd.read_csv(these_items[-1])
    tt = df1["TD_epoch_time_s_actual"][0]
    y = datetime.datetime.fromtimestamp(tt, local_tz)
    print(y)
    shift = int(str(y)[-6:-3])
    adhoc_timeshift = pd.Timedelta(shift, "hours")  # shift: winter -8, summer -7
    print("-> shift:", shift)

    for item in these_items:
        this_df = pd.read_csv(item)
        this_df["_datetime"] = (
            pd.to_datetime(this_df["TD_epoch_time_s_actual"], unit="s")
            + adhoc_timeshift
        )
        this_df.set_index("_datetime", inplace=True)
        df_gc = pd.concat([df_gc, this_df])
    print("-> GC csv file turned into pandas df_gc:")
    # print(df_gc)
    print(df_gc['TD_state_current_enum_actual'][100:110])

    # use
    STATES = list(set(df_gc["TD_state_current_enum_actual"]))
    h = lambda name: ds.sel(fitter_var=name).fitter_values
    key = "partial_fit_integral"
    KOL.resetCycle()
    (A, B), F = RMPL.Maker(size=(15, 6), grid=(2, 1), sharex=True)
    A.plot(ds._datetime, h(key), lw=0.7, c=KOL.getNext())
    RMPL.setLabels(A, "time", key, "Experiment Survey")

    Y = np.ones(df_gc.shape[0])
    for j, state in enumerate(STATES):
        msk = df_gc["TD_state_current_enum_actual"] == state
        B.scatter(df_gc.index[msk], Y[msk] * j, c=KOL.getNext())

    B.legend(ncols=2, fontsize=8, markerscale=2)
    saver(F, str(dirs.misc_results_folder / 'ExperimentSurvey.png'))
    # headless VM cannot show figures

    msk = df_gc['TD_state_current_enum_actual'] == TDSTATE_FOR_TRIGGER
    ref_times = df_gc.index[msk]
    print('->msk')
    print(msk[100:110])
    starts = ref_times[
        np.diff(
            [rt - ref_times[0] for rt in ref_times], prepend=pd.Timedelta(-10000, "min")
        )
        > pd.Timedelta(100, "min")
    ]
    ends = ref_times[
        np.diff(
            [rt - ref_times[0] for rt in ref_times], append=pd.Timedelta(10000, "min")
        )
        > pd.Timedelta(100, "min")
    ]

    PFI = h("partial_fit_integral").values

    timing_info = []
    count = 0
    for s, e in zip(starts, ends):  # iterate over reference periods
        _ref_msk = (ds._datetime >= s) & (ds._datetime <= e)
        _ref_times = ds._datetime[_ref_msk]

        # if combolog data does not cover range of gc log data
        if len(_ref_times) == 0:
            continue
        row = {}
        row["index"] = count
        count += 1
        row["ref_start"] = (_ref_times[0] + baseline_start_trim).values
        row["ref_end"] = (_ref_times[-1] - baseline_end_trim).values

        final_ref_msk = (ds._datetime >= row["ref_start"]) & (
            ds._datetime <= row["ref_end"]
        )
        final_ref_times = ds._datetime[final_ref_msk]
        if len(final_ref_times) > 0:
            row["ref_duration"] = (
                final_ref_times[-1] - final_ref_times[0]
            ).values.astype(np.double) * 1e-9
        else:
            row["ref_duration"] = 0

        row["start"] = (_ref_times[-1]).values + start_trim
        row["end"] = (_ref_times[-1] + chromatogram_duration).values - end_trim

        sample_msk = (ds._datetime >= row["start"]) & (ds._datetime <= row["end"])
        row["sample_points"] = sample_msk.values.sum()
        if row["sample_points"] <= 1:
            continue
        sample_times = ds._datetime[sample_msk]
        sample_interval = np.diff(
            (sample_times - sample_times[0]).values.astype(np.double) * 1e-9
        )

        row["mean_PFI"] = PFI[sample_msk].mean()
        row["max_data_gap"] = sample_interval.max()
        row["sample_duration_sec"] = (sample_times[-1] - sample_times[0]).values.astype(
            float
        ) * 1e-9
        BAD_REF_DURATION = row["ref_duration"] < ref_min_duration
        BAD_INTERVAL = sum(sample_interval > max_data_gap) > 0
        BAD_DURATION = (
            row["sample_duration_sec"] < chromatogram_duration.total_seconds() * 0.95
        )

        row["good"] = not (BAD_INTERVAL or BAD_DURATION or BAD_REF_DURATION)
        timing_info.append(row)

    df_time = pd.DataFrame(timing_info)    
    df_time.columns = df_time.columns.str.strip()  # added may help to suppress 'zero mask' error
    df_time.set_index("index", inplace=True)
    df_time.to_parquet(str(dirs.misc_results_folder / "chromatogram_times.parquet"))
    print("-> df time info: check if column 'good' has value true")
    print(df_time[:10])

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
    print("-> zero_mask")
    print(zero_mask)

    grouper = xr.where(
        zero_mask, 0, 1
    )  # define background by data before the start time
    zero_ref_ds = run_ds.zero_reference.zero_reference(
        transition_variable=grouper,
        reference_value=0,
        event_window_width=NUM_REFS_TO_AVE,
    )

    df_dsstats = pd.DataFrame(
        ds.fitter_values.mean(dim="_datetime").to_pandas(), columns=["mean"]
    )
    df_dsstats["std"] = ds.fitter_values.std(dim="_datetime").values
    df_dsstats.to_csv(dirs.misc_results_folder / "dataset_stats.csv")

    A, F = RMPL.Maker(size=(14, 6))
    key = "partial_fit_integral"

    Z = zero_ref_ds.sel(fitter_var=key).fitter_values
    cz = RMPL.cmap()
    Nrows = df_time.shape[0]
    for indx, row in df_time.iterrows():
        if not row["good"]:
            continue

        rmask = (zero_ref_ds._datetime >= row["ref_start"]) & (
            zero_ref_ds._datetime <= row["ref_end"]
        )
        rT = zero_ref_ds._datetime[rmask]

        this_start = min([row["start"], row["ref_start"]])
        this_end = max([row["end"], row["ref_end"]])
        dmask = (zero_ref_ds._datetime >= this_start - plot_pad) & (
            zero_ref_ds._datetime <= this_end + plot_pad
        )
        T = zero_ref_ds._datetime[dmask]

        sample_mask = T >= row["start"]
        T_sample_start = T[sample_mask][0]
        T = (T - T_sample_start) / 1e9
        Zmsk = Z[dmask]
        ave_params = {"_datetime": 10}
        A.plot(
            T.rolling(ave_params).mean(),
            Zmsk.rolling(ave_params).mean(),
            c=cz(indx / Nrows),
            lw=1,
            alpha=0.8,
        )

    A.set_yscale("symlog", linthresh=100)
    R_START = ((rT[0] - T_sample_start)).values * 1e-9
    R_END = ((rT[-1] - T_sample_start)).values * 1e-9

    ylimmy = A.get_ylim()
    A.plot([R_START, R_START], ylimmy, c="#aa0000", ls="--", lw=1.5, label="ref start")
    A.plot([R_END, R_END], ylimmy, c="#0000aa", ls="--", lw=1.5, label="ref end")
    A.legend(loc=1)
    A.set_ylim(ylimmy)

    RMPL.setLabels(A, "retention time [sec]", key, "Chromatogram Summary")
    saver(F, str(dirs.misc_results_folder / "ChromatogramSummary.png"))

    A, F = RMPL.Maker(size=(9, 5))
    key = "CHROM_OvenTemperature_Float_C_actual"
    Z = df_gc[key]
    cz = RMPL.cmap()
    for indx, row in df_time.iterrows():
        this_start = min([row["start"], row["ref_start"]])
        this_end = max([row["end"], row["ref_end"]])
        dmask = (df_gc.index >= this_start - plot_pad) & (
            df_gc.index <= this_end + plot_pad
        )
        T = df_gc.index[dmask]
        T = (T - row["start"]) * 1e-9
        A.plot(T, Z[dmask], c=cz(indx / Nrows), lw=1.5, alpha=0.5)
    RMPL.setLabels(
        A, "retention time [sec]", key, "Chromatogram Summary - Oven Temperature"
    )
    saver(F, str(dirs.misc_results_folder / "ChromatogramTemperatureSummary.png"))

    # find columns that vary:
    vary_columns = []
    for col in ds.fitter_var.values:
        z = ds.sel(fitter_var=col).fitter_values.values.std()
        if z > 1e-15:
            vary_columns.append(str(col))

    # Create summary plots of all non-constant columns in combologs
    this_saver = RMPL.SaveFigure(dirs.misc_results_folder / "SurveyPlots").savefig

    g = lambda col, msk: ds.sel(fitter_var=col).fitter_values[msk]
    for col in vary_columns:
        A, F = RMPL.Maker(size=(8, 4))
        count = 0
        for index, row in df_time.iterrows():
            if not row["good"]:
                continue
            count += 1
            this_mask_ref = (ds._datetime >= row["ref_start"]) & (
                ds._datetime < row["ref_end"]
            )
            this_mask_sample = (ds._datetime >= row["start"]) & (
                ds._datetime < row["end"]
            )

            A.plot(
                ds._datetime[this_mask_ref],
                g(col, this_mask_ref),
                c=KOL[0],
                zorder=5000,
                lw=2,
                label="reference" if count == 1 else "",
            )
            A.plot(
                ds._datetime[this_mask_sample],
                g(col, this_mask_sample),
                c=KOL[1],
                lw=1,
                label="sample" if count == 1 else "",
            )
        A.legend()
        A.tick_params(axis="x", rotation=30)
        RMPL.setLabels(A, "date", col, "", fontsize=10)
        this_saver(F, f"Survey_{col}.png", dpi=150)

    g = lambda col, msk: df_gc[col][msk]
    vary_gc_columns = []

    data_stats = []
    for col in df_gc.columns:
        try:
            Z = g(col, df_gc.index > df_gc.index[0])
            if Z.std() > 1e-15:
                vary_gc_columns.append(col)

            row = {"column": col, "mean": Z.mean(), "std": Z.std()}
            data_stats.append(row)
        except:
            print(col, "no good")

    df_gc_stats = pd.DataFrame(data_stats)
    df_gc_stats.to_csv(dirs.misc_results_folder / "GC_log_stats.csv")

    for col in vary_gc_columns:
        A, F = RMPL.Maker(size=(8, 4))
        count = 0
        for index, row in df_time.iterrows():
            if not row["good"]:
                continue
            count += 1
            this_mask_ref = (df_gc.index >= row["ref_start"]) & (
                df_gc.index < row["ref_end"]
            )
            this_mask_sample = (df_gc.index >= row["start"]) & (
                df_gc.index < row["end"]
            )

            A.plot(
                df_gc.index[this_mask_ref],
                g(col, this_mask_ref),
                c=KOL[0],
                zorder=5000,
                lw=2,
                label="reference" if count == 1 else "",
            )
            A.plot(
                df_gc.index[this_mask_sample],
                g(col, this_mask_sample),
                c=KOL[1],
                lw=1,
                label="sample" if count == 1 else "",
            )
        A.legend()
        A.tick_params(axis="x", rotation=30)
        RMPL.setLabels(A, "date", col, "", fontsize=10)
        this_saver(F, f"GC_Survey_{col}.png", dpi=150)

    t = int(time.time() - t0)
    print("\n*********************************** ")
    print("**  Step  2 done, took %.2f min ***" % (t / 60))
    return t


# ####### Step06_SurveyFitResultsAverageAndCorr
def retrieve_results(exp_subfolder, solver_name):
    if not exp_subfolder.exists():
        raise ValueError("No data found")

    results_folder = exp_subfolder / "results"
    results_txt_file = results_folder / f"{solver_name}.txt"
    convert_to_pd = lambda block: pd.read_table(StringIO(block), delimiter="\s+")
    if results_txt_file.exists():
        # print("Reading existing results")
        with open(results_txt_file, "r") as file:
            block = file.read()
        conc_table, score_table = block.split("Concentrations")[1].split("Scores")
        if "mean" not in conc_table:  # for correlation plot results
            conc_table = "        mean\n" + conc_table
        df_conc = convert_to_pd(conc_table)
        df_scores = convert_to_pd(score_table)
    else:
        return None, None

    q_file = results_folder / f"{solver_name}_Q_values.csv"
    if q_file.exists():
        df_q = pd.read_csv(q_file)
        df_q.set_index("CID", drop=True, inplace=True)
    df_conc["Q"] = 0
    for cid, qvalue in df_q.iterrows():
        df_conc.loc[cid, "Q"] = qvalue.iloc[0]

    return df_conc, df_scores


def time_corr(Q, T):
    if len(T) == 1:
        return np.array([0.000001])
    factor = np.zeros(Q.shape)
    for i in range(len(Q)):
        if i == 0:
            check = [1]
        elif i == len(Q) - 1:
            check = [i - 1]
        else:
            check = [i - 1, i + 1]
        DTinv = np.array([1 / abs(T[i] - T[c]) ** 2 for c in check])
        DTmean = DTinv.mean()
        factor[i] = DTmean
    return factor


def notebook6():
    print("********** start step 6: ********** SurveyFitResultsAverageAndCorr ***")
    t0 = time.time()

    saver = RMPL.SaveFigure().savefig

    # constants and parameters
    config = NAU.load_analysis_yaml()
    dirs = NAU.AnalysisPaths(config["paths"])
    allow_neg = {
        "CORR": config["correlation_spectra"]["allow_negatives_in_hunt"],
        "AVE": config["average_spectra"]["allow_negatives_in_hunt"],
    }
    chromatogram_duration = 60 * config["chromatogram_params"]["chromatogram_duration"]

    whichone = "AVE"
    fit_name = "Negs" if allow_neg[whichone] else "PosOnly"
    data_folder = (
        dirs.correlation_spectra_folder
        if whichone == "CORR"
        else dirs.average_spectra_folder
    )
    subfolders = [p for p in data_folder.iterdir() if p.is_dir()]
    subfolders.sort(key=lambda s: s.name)
    plot_dict = {}
    for subfolder in subfolders:
        if "Figures" in subfolder.name:
            continue
        tsec = int(subfolder.name.split("_")[1])
        df_conc, df_scores = retrieve_results(subfolder, f"R3_{fit_name}")
        if df_conc is None:
            continue
        for cid, row in df_conc.iterrows():
            if cid not in plot_dict:
                plot_dict[cid] = collections.defaultdict(list)
            udata = plot_dict[cid]
            udata["t_sec"].append(tsec)
            udata["Q"].append(row["Q"])
            udata["conc"].append(row["mean"])
            if whichone != "CORR":
                udata["sigma"].append(row["std"])

    found = list(plot_dict.keys())

    def sortguy(e):
        Q = np.array(plot_dict[e]["Q"])
        T = np.array(plot_dict[e]["t_sec"])
        u = np.exp((Q - 1) / 0.05) * time_corr(Q, T).mean()
        msk = T < 4000
        return u[msk].sum()

    found.sort(key=lambda e: np.sum(np.array(plot_dict[e]["Q"]) ** 2), reverse=True)
    found.sort(key=sortguy, reverse=True)

    NPLOTS = 4
    for j, fcid in enumerate(found[0 : 6 * 25]):
        udata = plot_dict[fcid]
        z = j % NPLOTS
        if z == 0:
            A, F = RMPL.Maker(grid=(2, NPLOTS), size=(5 * NPLOTS, 5))
        A[z].scatter(udata["t_sec"], udata["conc"], c=KOL[0], s=5)
        A[z + NPLOTS].scatter(udata["t_sec"], udata["Q"], c=KOL[1], s=5)
        if z == 0:
            A[z].set_ylabel("conc [ppb]", fontsize=10)
            A[z + NPLOTS].set_ylabel("Q", fontsize=10)
        A[z + NPLOTS].set_xlabel("time [sec]", fontsize=10)
        A[z].set_title("%s" % PubChemEntry(fcid), fontsize=12)
        if z + NPLOTS == 2 * NPLOTS - 1:
            saver(F)
            break

    fig_folder = data_folder / "_Figures"
    fig_folder.mkdir(exist_ok=True)
    saver = RMPL.SaveFigure(fig_folder).savefig
    mongo = get_mongo_spectral_library_client()
    NPLOTS = 1
    for j, fcid in enumerate(found[:]):
        if fcid == 0:
            continue
        Q = np.array(plot_dict[fcid]["Q"])
        T = np.array(plot_dict[fcid]["t_sec"])
        factor = time_corr(Q, T)
        sizes = 600 * factor + 1
        pubby = mongo.extendedPubChemEntries.find_one({"cid": fcid})
        smiles = pubby["names"]["canonical_SMILES"]
        mol = struct.get_mol_from_SMILES(smiles)
        udata = plot_dict[fcid]
        z = j % NPLOTS
        if z == 0:
            A, F = RMPL.Maker(grid=(2, NPLOTS), size=(7, 5))
        A[z].scatter(udata["t_sec"], udata["conc"], c=KOL[0], s=sizes)
        A[z + NPLOTS].scatter(udata["t_sec"], udata["Q"], c=KOL[1], s=sizes)
        if z == 0:
            A[z].set_ylabel("conc [ppb]", fontsize=10)
            A[z + NPLOTS].set_ylabel("Q", fontsize=10)
            A[z].set_xlim([0, chromatogram_duration])
            A[z + NPLOTS].set_xlim([0, chromatogram_duration])
        A[z + NPLOTS].set_xlabel("time [sec]", fontsize=10)
        A[z].set_title("%s" % PubChemEntry(fcid), fontsize=12)
        if z + NPLOTS == 2 * NPLOTS - 1:
            B = F.add_axes((0.85, 0.85, 0.12, 0.12))
            B.imshow(draw_molecule_no_Hs(mol, size=(300, 200)))
            B.set_xticks([])
            B.set_yticks([])
            saver(F, f"J%03d_{whichone}_Plot_%d.png" % (j, fcid), dpi=300, closeMe=True)

    t = int(time.time() - t0)
    print("\n*********************************** ")
    print("**  Step  6 done, took %.2f min ***" % (t / 60))
    return t


# ####### Step07_HuntingDigester
def get_Q_prob(Q, N, extend_to_Q_below_1=True):
    """args:
      Q: Q for a given compound in a fit results
      N: total number of compounds found
      returns the probability that the found compound was really in the gas sample

      Valid for R3 (running four steps where the Q threshold is lowered with each step)"""

    N = np.clip(N, 1, 55)

    if extend_to_Q_below_1:
        lowQ = np.ones(Q.shape)
        lowQ[Q < 1] = Q[Q < 1]

        Q = np.clip(Q, 1, np.inf)
    else:
        print("Q less than 1 not allowed")
        return None

    A = 0.31407 + 0.202362 / N - 0.00858516 * N + 0.000113417 * N ** 2
    B = 3.20283 - 0.0432235 * N + 0.00115416 * N ** 2
    C = 2.11314 - 0.0687511 * N + 0.000622635 * N ** 2

    rho = -B * (np.log10(Q - 1) + C)
    f = A + (1 - A) / (1 + np.exp(rho))
    if extend_to_Q_below_1:
        f = lowQ * f
    return f


def weighted_mean(X, W):
    return (X * W).sum() / W.sum()


def calc_prob(dframe):
    for key in [
        "q_prob",
        "bayes_prob",
        "peak_len",
        "q_sum",
        "pos_conc",
        "mean_conc",
        "T_center",
        "T_width",
        "T_start",
        "T_end",
    ]:
        dframe[key] = 0

    for j, (_, row) in enumerate(dframe.iterrows()):
        Q = row["Q"]
        find_count = row["find_count"]
        T = row["T"]
        conc = row["conc"]
        prob = get_Q_prob(Q, find_count)
        agg_prob = 0 if row["cluster"] == -1 else 1 - np.prod(1 - prob)

        dframe["q_prob"].iloc[j] = agg_prob

        dframe["peak_len"].iloc[j] = len(T)
        dframe["q_sum"].iloc[j] = (Q).sum()
        dframe["pos_conc"].iloc[j] = np.all(conc >= 0)

        dframe["mean_conc"].iloc[j] = weighted_mean(conc, Q)
        T_0 = weighted_mean(T, Q)
        dframe["T_center"].iloc[j] = T_0
        dframe["T_width"].iloc[j] = 2 * np.sqrt(weighted_mean((T - T_0) ** 2, Q))
        dframe["T_start"].iloc[j] = T.min()
        dframe["T_end"].iloc[j] = T.max()

        full_t = np.arange(T.min(), T.max() + row["T_step"], row["T_step"])
        evidence = np.zeros(full_t.shape)
        found_msk = np.array([t_i in row["T"] for t_i in full_t])
        try:
            evidence[found_msk] = get_Q_prob(row["Q"], row["find_count"])
        except:
            set_trace()
        evidence[~found_msk] = np.nan
        bayes_result = run_bayes_on_evidence(evidence)
        dframe["bayes_prob"].iloc[j] = bayes_result.iloc[-1]["present"]

    return dframe


def conditional_prob(h, e, pTHRESH=0.8):
    if np.isnan(e):
        EPSILON = 0.01
        if h == "present":
            cprob = 0.5 - EPSILON
        elif h == "absent":
            cprob = 0.5 + EPSILON
    else:
        if e < 0.5:
            if h == "present":
                cprob = 0.5
            elif h == "absent":
                cprob = 0.5
        elif 0.5 <= e < pTHRESH:
            DELTA = 0.05
            if h == "present":
                cprob = 0.5 + DELTA * e
            elif h == "absent":
                cprob = 0.5 - DELTA * e
        else:
            if h == "present":
                cprob = e
            elif h == "absent":
                cprob = 1 - e
    return cprob


def compute_posterior(hypotheses, evidence, prior, pTHRESH=0.25):
    posterior = np.zeros(prior.shape)
    for j, h in enumerate(hypotheses):
        posterior[j] = conditional_prob(h, evidence, pTHRESH=pTHRESH) * prior[j]
    N = len(posterior)
    posterior /= posterior.sum()
    REG = 1e-6
    posterior += REG
    posterior /= posterior.sum()
    return posterior


def run_bayes_on_evidence(evidence_list):
    HYPOS = ["present", "absent"]

    prior = np.array([0.5, 0.5])
    current_prob = prior
    post_data = [{h: p for h, p in zip(HYPOS, prior)}]
    post_data = []
    for n, e in enumerate(evidence_list):
        posterior = compute_posterior(HYPOS, e, current_prob, pTHRESH=0.85)
        current_prob = posterior
        post_data.append({h: p for h, p in zip(HYPOS, posterior)})
    return pd.DataFrame(post_data)


def notebook7():
    print("********** start step 7: ********** ")
    t0 = time.time()

    # constants and parameters
    config = NAU.load_analysis_yaml()
    dirs = NAU.AnalysisPaths(config["paths"])
    allow_neg = {
        "CORR": config["correlation_spectra"]["allow_negatives_in_hunt"],
        "AVE": config["average_spectra"]["allow_negatives_in_hunt"],
    }
    chromatogram_duration = config["chromatogram_params"]["chromatogram_duration"] * 60
    cluster_param = config["peak_clustering"]

    data_folders = {}
    data_folders["AVE"] = dirs.average_spectra_folder
    data_folders["CORR"] = dirs.correlation_spectra_folder
    param_name = {"AVE": "average_spectra", "CORR": "correlation_spectra"}

    dfs = pd.DataFrame()
    for EXPT in ["AVE", "CORR"]:
        params = config[param_name[EXPT]]
        fit_name = "Negs" if allow_neg[EXPT] else "PosOnly"
        data_folder = data_folders[EXPT]
        subfolders = [p for p in data_folder.iterdir() if p.is_dir()]
        subfolders.sort(key=lambda s: s.name)

        plot_dict = {}
        for subfolder in subfolders:
            if "Figures" in subfolder.name:
                continue
            tsec = int(subfolder.name.split("_")[1])
            df_conc, df_scores = retrieve_results(subfolder, f"R3_{fit_name}")
            if df_conc is None:
                continue
            for cid, row in df_conc.iterrows():
                if cid == 0:
                    continue
                if cid not in plot_dict:
                    plot_dict[cid] = collections.defaultdict(list)
                udata = plot_dict[cid]
                udata["t_sec"].append(tsec)
                udata["Q"].append(row["Q"])
                udata["conc"].append(row["mean"])
                udata["find_count"].append(df_conc.shape[0] - 1)
                if "std" in row:
                    udata["sigma"].append(row["std"])

        result_list = []
        for cid, U in plot_dict.items():
            if cid == 0:
                print("huh")
            U2D = [[u, 0] for u in U["t_sec"]]
            eps = params["step_size_sec"] * (
                cluster_param["max_gap_for_peak_in_steps"] + 0.5
            )
            this_cluster = DBSCAN(
                eps=eps, min_samples=cluster_param["min_num_samples_in_peak"]
            ).fit(U2D)
            for label in set(this_cluster.labels_):
                get_me = lambda name: np.array(U[name])[this_cluster.labels_ == label]
                res = {}
                res["T"] = get_me("t_sec")
                res["dT"] = np.diff(
                    res["T"], prepend=res["T"][0] - params["step_size_sec"]
                )
                res["Q"] = get_me("Q")
                res["conc"] = get_me("conc")
                res["cid"] = cid
                res["cluster"] = label
                res["find_count"] = get_me("find_count")
                res["T_step"] = params["step_size_sec"]
                result_list.append(res)
        df = pd.DataFrame(result_list)
        df.set_index("cid", drop=True, inplace=True)
        df["expt"] = EXPT
        dfs = pd.concat([dfs, df])

    def q_prob_max(dframe):
        dframe.sort_values(["q_prob"], ascending=[False], inplace=True)
        return dframe.iloc[0]

    sorted_cids = {}

    df_prob = dfs.groupby("cid").apply(calc_prob)
    df_prob = df_prob.droplevel(1)

    prob_to_use = cluster_param["prob_to_use"]
    mask = df_prob[prob_to_use] >= cluster_param["overall_probability_threshold"]
    mask &= df_prob["pos_conc"]
    best_cids = df_prob[mask].groupby("cid").apply(q_prob_max)
    best_cids.sort_values("T_center", inplace=True, ascending=True)
    print("best_cids: ", best_cids)

    count = 0
    for tcid in best_cids.index:
        if tcid not in sorted_cids.keys():
            sorted_cids[tcid] = count
            count += 1

    cz = RMPL.cmap("plasma")

    def mapcolor(value, p0=0.85, alpha=0.5):
        z = ((1 - value) / (1 - p0)) ** alpha
        return cz(z)

    A, F = RMPL.Maker(size=(14, 17))
    for EXPT in ["AVE", "CORR"]:
        this_df_prob = df_prob[df_prob["expt"] == EXPT]
        for j, (cid, best) in enumerate(this_df_prob.iterrows()):
            if cid not in sorted_cids:
                continue
            if not best["pos_conc"]:
                continue
            p = best[prob_to_use]
            if p < cluster_param["overall_probability_threshold"]:
                continue

            J = sorted_cids[cid]
            offset = -0.2 if EXPT == "AVE" else 0.2
            ones = np.ones(best["T"].shape)
            A.plot([0, chromatogram_duration], [J, J], c="#aaaaaa44", lw=0.5, zorder=-1)
            A.plot(
                best["T"],
                ones * (J + offset),
                c=mapcolor(p, p0=cluster_param["overall_probability_threshold"]),
                alpha=1,
                lw=3,
            )
            if not best["pos_conc"]:
                A.scatter(
                    best["T"].mean(),
                    J + offset,
                    fc="#ffffff",
                    ec="none",
                    s=8,
                    zorder=5000,
                )
            A.annotate(
                PubChemEntry(cid), xy=(-30, J), ha="right", va="center", fontsize=9.5
            )
    ylim = A.get_ylim()
    A.set_ylim(ylim[::-1])
    A.set_xlim([-0.5 * chromatogram_duration, chromatogram_duration])
    RMPL.setLabels(
        A, "chromatogram time [sec]", "compounds found", "Compound Hunting Summary"
    )
    saver(F, str(dirs.misc_results_folder / "CompoundHuntingSummary.png"))

    best_cids.to_parquet(dirs.misc_results_folder / "HighestRankedClusters.parquet")
    df_prob.to_parquet(dirs.misc_results_folder / "AllClusters.parquet")
    test_df = pd.read_parquet(
        dirs.misc_results_folder / "HighestRankedClusters.parquet"
    )
    print(test_df)

    t = int(time.time() - t0)
    print("\n*********************************** ")
    print("**  Step  7 done, took %.2f min ***" % (t / 60))
    return t


if __name__ == "__main__":
    notebook1()
    # notebook2()
    # notebook6()
    # notebook7()
