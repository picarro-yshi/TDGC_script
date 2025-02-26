# TDGC analysis, functions for notebooks no need sub-tasking
# corresponding to Chris's notebook 8,10,11,14

import time
import numpy as np
import xarray as xr
import pandas as pd

pd.set_option("display.max_rows", 300)
import collections
import pickle

import warnings

warnings.filterwarnings("ignore")

import NotebookAnalysisUtilities as NAU
import DataUtilities3.MPL_rella as RMPL
import DataUtilities3.RellaColor as RC

RMPL.SetTheme("EIGHT")
saver = RMPL.SaveFigure().savefig
KOL = RC.SeventiesFunk(brtValue=0.25)
KOLmodel = RC.SeventiesFunk(brtValue=0.25)

from SpectroscopyTools.HitranDBModule import PubChemEntry
from spectral_arithmetic.linear_least_squares import least_squares
from spectral_toolkit.compound_search.r_cubed.basic.r_cubed_rev_2 import Allan

from td_analysis.peak_fitting.process_event_data import _rmse_results
from td_analysis.peak_fitting.peaks_find import find_guess_peaks
from td_analysis.peak_fitting.peaks_fit import fit_data
from td_analysis.peak_fitting.skewed_gaussian_models import (
    SkewedGaussian,
    MultiSkewedGaussian,
)


# ####### Step08_CreateFitRecipesInTime
def notebook8():
    print("********** start step 8: ********** CreateFitRecipesInTime ***")
    t0 = time.time()

    # constants and parameters
    config = NAU.load_analysis_yaml()
    dirs = NAU.AnalysisPaths(config["paths"])
    df_prob = pd.read_parquet(dirs.misc_results_folder / "AllClusters.parquet")

    # make fitter list
    PROB = config["peak_clustering"]["overall_probability_threshold"]
    KNEE = config["refitting_params"]["early_time_knee"]
    chromatogram_duration = config["chromatogram_params"]["chromatogram_duration"] * 60

    fit_recipes = {}
    A, F = RMPL.Maker(size=(8, 5))

    for J, MIN_EXTRA_T in enumerate([30, 60, 120, 180, 300, 450, 600]):
        fit_cids = collections.defaultdict(list)
        for EXPT in ["AVE", "CORR"]:
            this_df_prob = df_prob[df_prob["expt"] == EXPT]
            for j, (cid, best) in enumerate(this_df_prob.iterrows()):
                if not best["pos_conc"]:
                    continue
                p = best["q_prob"]
                if p < PROB:
                    continue

                if best["T_start"] <= KNEE:
                    max_fitstart = 0
                elif KNEE < best["T_start"] <= 2 * KNEE:
                    max_fitstart = (best["T_start"] - KNEE) * 2
                else:
                    max_fitstart = best["T_start"]

                T_range = best["T_end"] - best["T_start"]

                T_pad = T_range if T_range > MIN_EXTRA_T else MIN_EXTRA_T

                fitstart = best["T_start"] - T_pad
                fitstart = min([fitstart, max_fitstart])

                fitend = best["T_end"] + T_pad
                if cid == 6334:
                    print(fitstart, fitend)
                DT = best["T_step"]
                for tbin in range(fitstart, fitend + DT, DT):
                    if (tbin >= 0) & (tbin <= chromatogram_duration):
                        if cid not in fit_cids[tbin]:
                            fit_cids[tbin].append(cid)
        fit_recipes[MIN_EXTRA_T] = fit_cids
        for t, cids in fit_cids.items():
            labby = "pad time = %d sec" % MIN_EXTRA_T if (t == 300) else ""
            A.scatter(t, len(cids), c=KOL[J], label=labby, s=10)
    A.legend()
    RMPL.setLabels(A, "time [sec]", "# CIDS", "# Compounds in Fit")
    saver(F, str(dirs.misc_results_folder / "NumCompoundsInFitExclude.png"), dpi=400)

    data_rows = []
    hash_list = []

    add_manual = {}  # {887: (0, 500)}
    for MIN_EXTRA_T, fit_cids in fit_recipes.items():
        key_list = list(fit_cids.keys())
        key_list.sort()
        for tbin in key_list:
            fit_list = fit_cids[tbin]
            for mcid, (b0, b1) in add_manual.items():
                if b0 <= tbin <= b1:
                    fit_list.append(mcid)
            if fit_list:
                fit_list.sort()
            recipe_hash = hash(tuple(fit_list))
            if recipe_hash not in hash_list:
                hash_list.append(recipe_hash)
            recipe_id = hash_list.index(recipe_hash)
            this_row = {"tbin": tbin, "t_ave": MIN_EXTRA_T, "recipe_id": recipe_id}

            for cid in fit_list:
                this_row.update({cid: 1})
            data_rows.append(this_row)

    df_recipes = pd.DataFrame(data_rows)
    df_recipes.set_index(["tbin", "t_ave"], inplace=True)
    df_recipes.sort_index(inplace=True)
    cols = df_recipes.columns.to_list()
    numcols = cols[1:]
    numcols.sort()
    cols = [cols[0]] + numcols
    df_recipes = df_recipes[cols]
    df_recipes.to_parquet(dirs.misc_results_folder / "fit_cid_table_exclude.parquet")
    mapper = {col: int(col) for col in df_recipes.columns if col != "recipe_id"}
    df_recipes.rename(columns=mapper, inplace=True)
    print(df_recipes)

    X = df_recipes.loc[(slice(None), 120), :]
    for tbin in range(0, 90, 10):
        Y = X.loc[tbin].dropna(axis=1)
        print(tbin, Y["recipe_id"].values, Y.columns.to_list())

    t = int(time.time() - t0)
    print("\n*********************************** ")
    print("**  Step  8 done, took %.2f min ***" % (t / 60))
    return t


# ####### Step10_CreateLSQSummaryFile
def notebook10():
    print("********* start step 10: ********** CreateLSQSummaryFile ***")
    t0 = time.time()

    # constants and parameters
    config = NAU.load_analysis_yaml()
    dirs = NAU.AnalysisPaths(config["paths"])

    ds = xr.open_zarr(dirs.zarr_path)
    ds.close()
    print(ds)

    fom_list = []
    for item in dirs.refit_folder.glob("ref*"):
        dframe_ref = pd.read_parquet(item)
        recipe_id = int(dframe_ref["recipe_id"].median())
        dframe_ref.drop("recipe_id", axis=1, inplace=True)
        row = {}
        row["recipe_id"] = recipe_id

        for col in dframe_ref.columns:
            if col in ["_datetime", "recipe_id"]:
                continue
            try:
                astd = Allan(dframe_ref[col].values[5000:8000])
                row[col] = astd.S[0] * 3
            except:
                # print("index error: ", col)
                pass
        fom_list.append(row)
    ref_results_lsq = pd.DataFrame(fom_list)
    print(ref_results_lsq)

    results_lsq = pd.DataFrame()
    for item in dirs.refit_folder.glob("sample*"):
        dframe = pd.read_parquet(item)
        results_lsq = pd.concat([results_lsq, dframe])

    for name, res_df in zip(
        ["_results_lsq.parquet", "_ref_results_lsq.parquet"],
        [results_lsq, ref_results_lsq],
    ):
        res_df.to_parquet(dirs.refit_folder / name)

    ref_results_lsq_full = pd.DataFrame()

    for item in dirs.refit_folder.glob("ref*"):
        dframe = pd.read_parquet(item)
        ref_results_lsq_full = pd.concat([ref_results_lsq_full, dframe])

    ref_results_lsq_full.to_parquet(dirs.refit_folder / "_ref_results_lsq_full.parquet")

    t = int(time.time() - t0)
    print("\n*********************************** ")
    print("**  Step 10 done, took %.2f min ***" % (t / 60))
    return t


# ####### Step10.1_PlotLSQResults
def notebook11():
    print("********* start step 11: ********** ")
    t0 = time.time()

    # constants and parameters
    config = NAU.load_analysis_yaml()
    dirs = NAU.AnalysisPaths(config["paths"])

    # Get data from zarr store
    ds = NAU.load_ds_from_zarr(dirs.zarr_path)

    expt_mask = (ds._datetime >= config["times"]["expt_start"]) & (
        ds._datetime <= config["times"]["expt_end"]
    )
    ds = ds.sel(_datetime=expt_mask)
    print(ds)

    # load chromatogram transitions
    df_time = pd.read_parquet(dirs.misc_results_folder / "chromatogram_times.parquet")
    print(df_time)

    results_lsq = pd.read_parquet(dirs.refit_folder / "_results_lsq.parquet")
    results_lsq.sort_index(inplace=True)
    ref_results_lsq = pd.read_parquet(dirs.refit_folder / "_ref_results_lsq.parquet")
    print(ref_results_lsq)

    fom_df = pd.DataFrame()
    for rid, dframe in results_lsq.groupby("recipe_id"):
        msk = ref_results_lsq["recipe_id"] == rid
        three_sig = ref_results_lsq[msk]
        dframe.dropna(axis=1, inplace=True)
        three_sig.dropna(axis=1, inplace=True)
        for col in three_sig:
            dframe[col] /= three_sig[col].values[0]
        fom_df = pd.concat([fom_df, dframe])
    fom_df.sort_index(inplace=True)

    KOL2 = RC.KawaiiPunchyCute(brtValue=0.0)
    KOL2.make_long(brt_list=[-0.5, -0.2])

    ROLLING = 5
    for expt_name, plot_df in zip(["FOM", "CONC"], [fom_df, results_lsq]):
        cols = list(plot_df.columns)
        cols.sort()
        pfi = ds.sel(fitter_var="partial_fit_integral").fitter_values
        pfi0 = pfi.values - pfi.values[0:20].mean()
        count = 0
        plot_pad = pd.Timedelta(5, "min")

        save_dir = dirs.refit_folder / "Figures"
        save_dir.mkdir(exist_ok=True, parents=True)

        scale = np.sqrt(ROLLING) if expt_name == "FOM" else 1
        for j, key in enumerate(cols):
            print(expt_name, key)
            if "recipe" in key:
                continue
            A, F = RMPL.Maker(size=(18, 10), grid=(6, 9))
            dat = plot_df[key].dropna()
            for i, trow in df_time.iterrows():
                start = trow["start"] - plot_pad
                stop = trow["end"] + plot_pad
                pfi_msk = (pfi._datetime >= start) & (pfi._datetime <= stop)
                A[i].plot(
                    pfi._datetime[pfi_msk],
                    pfi0[pfi_msk] / pfi0.max() * dat.values.max(),
                    c="#444444",
                    lw=1,
                    zorder=150,
                )

                dat_msk = (dat.index >= start) & (dat.index <= stop)
                A[i].plot(
                    dat.index[dat_msk],
                    dat[dat_msk].rolling(window=ROLLING).mean() * scale,
                    c=KOL2[j],
                    zorder=100,
                    lw=1.25,
                )
                A[i].annotate(
                    dat.index[dat_msk][0],
                    xy=(0.03, 0.05),
                    xycoords="axes fraction",
                    fontsize=7,
                    color="#888888",
                )

            mx = dat.values.max() * scale
            mn = dat.values.min() * scale
            for k, a in enumerate(A):
                a.set_ylim([mn, mx])
                a.set_xticklabels([])
                if k % 9 != 0:
                    a.set_yticklabels([])
                else:
                    if expt_name == "FOM":
                        labby = "FOM"
                    else:
                        if "gasConcs" in key:
                            labby = "conc [ppb]"
                        elif "base_0" in key:
                            labby = "base0 [ppb/cm]"
                        elif "base_1" in key:
                            labby = "base1 [ppb/cm per wvn]"
                    a.set_ylabel(expt_name if expt_name == "FOM" else "conc [ppb]")

            if "gasConcs" in key:
                cid = int(key.split("_")[-1])
                name = PubChemEntry(cid)
            else:
                cid = key
                name = key

            F.suptitle(f"{name} rolling ave = {ROLLING}", fontsize=24)
            fn = save_dir / f"{expt_name}_REFIT_{cid}.png"

            saver(F, str(fn), dpi=300, closeMe=False)
            count += 1

            if count == 1:
                break

    t = int(time.time() - t0)
    print("\n*********************************** ")
    print("**  Step 11 done, took %.2f min ***" % (t / 60))
    return t


# ####### Step14_FitConcentrationPeaks
def notebook14():
    print("********* start step 14: ********** FitConcentrationPeaks ***")
    t0 = time.time()

    # constants and parameters
    config = NAU.load_analysis_yaml()
    dirs = NAU.AnalysisPaths(config["paths"])

    MAX_GAP = (
        config["peak_clustering"]["max_gap_for_peak_in_steps"]
        * config["average_spectra"]["step_size_sec"]
    )
    SCALING = config["peak_fitting_params"]["peak_scaling"]

    ds = xr.open_zarr(dirs.zarr_path)
    ds.close()
    print(ds)

    results_lsq = pd.read_parquet(dirs.refit_folder / "_results_lsq.parquet")
    results_lsq.sort_index(inplace=True)
    ref_results_lsq = pd.read_parquet(dirs.refit_folder / "_ref_results_lsq.parquet")
    print(ref_results_lsq)

    # load chromatogram transitions
    df_time = pd.read_parquet(dirs.misc_results_folder / "chromatogram_times.parquet")
    print(df_time)

    def special_std(dframe):
        return dframe.std() / np.sqrt(dframe.shape[0] + 1)

    cols = list(results_lsq.columns)
    cols.sort()
    BIN_WIDTH = 2.5
    save_dir = dirs.refit_folder / "ConcPeakFit"
    save_dir.mkdir(exist_ok=True, parents=True)
    this_saver = RMPL.SaveFigure(DIR=save_dir).savefig

    def get_group(X):
        Y = np.array([int(round((x + 0.5 * BIN_WIDTH) / BIN_WIDTH)) for x in X])
        return Y

    mean_fit_results = {}
    DO_NOT_PULSE_FIT = [222]
    for j, key in enumerate(cols):
        if "recipe" in key or "base" in key or "rmse" in key:
            continue
        fit_me = True
        for cid in DO_NOT_PULSE_FIT:
            if f"lsq_gasConcs_{cid}" == key:
                fit_me = False
                print("skipping", key)
        if not fit_me:
            continue
        print(key)
        dat = results_lsq[key].dropna()
        dat_df = pd.DataFrame(dat)

        dat_df["delta_t"] = 0
        dat_df["time_group"] = 0
        dat_df["cycle"] = 0

        for i, trow in df_time.iterrows():
            start = trow["start"]
            stop = trow["end"]
            dat_msk = (dat_df.index >= start) & (dat_df.index <= stop)
            X = dat_df.index[dat_msk]
            X = (X - start).total_seconds()
            dat_df.loc[dat_msk, "delta_t"] = X
            dat_df.loc[dat_msk, "time_group"] = get_group(X)
            dat_df.loc[dat_msk, "cycle"] = np.ones(X.shape) * i
        mean_dat_df = dat_df.groupby("time_group").mean()
        std_dat_df = dat_df.groupby("time_group").apply(special_std)

        TmF = mean_dat_df["delta_t"].values
        I = np.arange(len(TmF))
        starts = I[np.diff(TmF, prepend=-2 * MAX_GAP) > MAX_GAP]
        ends = I[np.diff(TmF, append=TmF.max() + 2 * MAX_GAP) > MAX_GAP]
        YmF, Ystd = mean_dat_df[key].values, std_dat_df[key].dropna().values
        results_list = []
        for s, e in zip(starts, ends):
            (A, B), F = RMPL.MakerCal(size=(8, 9), N=3)
            A.plot(dat_df["delta_t"], dat_df[key], c=KOL.getNext(), lw=0.5, alpha=0.2)
            A.plot(
                mean_dat_df["delta_t"],
                mean_dat_df[key],
                lw=3,
                c=KOL.getNext(),
                zorder=10000,
            )
            B.plot(
                mean_dat_df["delta_t"],
                mean_dat_df[key],
                lw=3,
                c=KOL.getCurrent(),
                zorder=100,
                alpha=0.4,
            )

            results_dict = {}
            results_dict["start_time"] = TmF[s]
            results_dict["end_time"] = TmF[e]
            Tm = TmF[s:e]
            Ym = YmF[s:e]
            MIN_HT = (Ym.max() - Ym.min()) / 50
            MIN_WIDTH = 5
            LO = SkewedGaussian(amplitude=0, location=min(Tm), std_dev=3, skew=-5)
            HI = SkewedGaussian(
                amplitude=100000, location=max(Tm), std_dev=300, skew=10
            )
            BASE_STD = np.median(Ystd)
            SIG_STD = 4 * BASE_STD
            p0 = find_guess_peaks(
                Tm,
                Ym,
                MIN_WIDTH,
                MIN_HT,
                low_bound=LO,
                up_bound=HI,
                baseline_std=BASE_STD,
                signal_std=SIG_STD,
                max_peaks=3,
            )
            print(p0)
            results_dict["model"] = None
            if len(p0.gaussians) > 0:
                try:
                    fit_results = fit_data(Tm, Ym, p0, LO, HI, max_iter=50000)
                except ValueError:
                    print("FAILED")
                    continue
                rmse_values = _rmse_results(fit_results, Ym, Tm)
                best_peak_number = min(rmse_values, key=rmse_values.get)
                fit_result = fit_results[best_peak_number]
                bfit = MultiSkewedGaussian.from_list(list(fit_result["popt"]))
                bintegral = bfit.integrate((Tm[0], Tm[-1]))
                print(bfit.integrate((0, Tm.max())), bfit)
                results_dict["model"] = bfit

                A.plot(Tm, bfit.evaluate(Tm), lw=0.5, zorder=1000000, c="#ffffff")
                A.plot(
                    Tm,
                    bfit.evaluate(Tm),
                    lw=2,
                    zorder=1000000 - 5,
                    c="#000000",
                    label="fit (int=%.1f)" % bintegral,
                )
                KOLmodel.resetCycle()
                for g_id, gaussian in enumerate(bfit.gaussians):
                    integral = gaussian.integrate((Tm[0], Tm[-1]))
                    B.plot(
                        Tm,
                        gaussian.evaluate(Tm),
                        c=KOLmodel.getNext(),
                        lw=2,
                        label="peak %d (int=%.1f)" % (g_id + 1, integral),
                        zorder=10000,
                    )

            for a in [A, B]:
                a.set_xlim([Tm[0] - 20, Tm[-1] + 20])
            A.legend(fontsize=9)
            B.legend(fontsize=8)
            CID = int(key.split("_")[-1])
            RMPL.setLabels(
                A,
                "chromatogram time [sec]",
                "read conc. [ppb]",
                f"{str(PubChemEntry(CID))} ({TmF[s]:.0f}-{TmF[e]:.0f} sec)",
            )
            fn = f"MeanPulseFit_{key}_{TmF[s]:.0f}_{TmF[e]:.0f}_sec.png"
            this_saver(F, fn, dpi=300, closeMe=True)
            results_list.append(results_dict)
        mean_fit_results[key] = results_list

    # 135 items. the original cell is split, so kernel will not crash
    with open(save_dir / "SaveSkewedGuassians.pkl", "wb") as outpkl:
        pickle.dump(mean_fit_results, outpkl)

    with open(save_dir / "SaveSkewedGuassians.pkl", "rb") as inpkl:
        mean_fit_results = pickle.load(inpkl)

    def Amatrix(Alist):
        return np.array(Alist).T

    for key, results_list in mean_fit_results.items():
        CID = int(key.split("_")[-1])
        print(key)
        dat = results_lsq[key].dropna()
        for j, model_info in enumerate(results_list):
            if model_info["model"] is None:
                continue
            conc_dat_list = []
            for i, trow in df_time.iterrows():

                start = trow["start"]
                stop = trow["end"]
                dat_msk = (dat.index >= start) & (dat.index <= stop)
                X = dat.index[dat_msk]
                X = (X - start).total_seconds().values
                Y = dat[dat_msk].values

                conc_dict = {
                    "start": start,
                    "end": stop,
                }
                prefix = ""

                fstart = model_info["start_time"]
                fend = model_info["end_time"]
                sub_msk = (X >= fstart) & (X <= fend)
                thisX = X[sub_msk]
                thisY = Y[sub_msk]

                Alist = [np.ones(thisX.shape) / (fend - fstart)]
                for gaussian in model_info["model"].gaussians:
                    area = gaussian.integrate((fstart, fend))
                    Alist.append(gaussian.evaluate(thisX) / area)
                fit = least_squares(Amatrix(Alist), thisY)
                conc_dict[prefix + "start_sec"] = fstart
                conc_dict[prefix + "end_sec"] = fend
                for k, param in enumerate(fit[0]):
                    lbl = "offset" if k == 0 else f"conc{k}"
                    conc_dict[prefix + lbl] = param
                conc_dat_list.append(conc_dict)

            conc_df = pd.DataFrame(conc_dat_list)
            (A, B), F = RMPL.MakerSideBySide(size=(12, 5), N=3)
            KOLmodel.resetCycle()

            TO_PPT = 1000
            for i in range(0, 10):
                datkey = "conc%d" % i if i != 0 else "offset"
                if datkey in conc_df:
                    A.scatter(
                        conc_df["start"],
                        TO_PPT * conc_df[datkey] / SCALING,
                        c=KOLmodel.getNext(),
                        s=15,
                        label=datkey,
                    )
                    A.plot(
                        conc_df["start"],
                        TO_PPT * conc_df[datkey] / SCALING,
                        c=KOLmodel.getCurrent(),
                        lw=0.5,
                    )

            start_pk = conc_df["start_sec"][0]
            end_pk = conc_df["end_sec"][0]

            for u, gaussian in enumerate(model_info["model"].gaussians):
                tsim = np.linspace(start_pk, end_pk, 1000)
                B.plot(
                    tsim,
                    gaussian.evaluate(tsim),
                    c=KOL[u + 1],
                    label="peak %d" % (u + 1),
                )
            A.legend(fontsize=8)
            B.legend(fontsize=8)
            A.tick_params(axis="x", rotation=45)
            RMPL.setLabels(A, "date", "concentration [ppt]", PubChemEntry(CID))
            RMPL.lowRight(B)
            RMPL.setLabels(B, "chromatogram time [sec]", "read conc. [ppb]", "peaks")
            fn = f"ConcPlot_{CID}_{start_pk:.0f}_{end_pk:.0f}_sec"
            this_saver(F, fn, dpi=250)
            conc_df.to_parquet(
                save_dir / f"ConcData_{CID}_{start_pk:.0f}_{end_pk:.0f}_sec.parquet"
            )

    t = int(time.time() - t0)
    print("\n*********************************** ")
    print("**  Step 14 done, took %.2f min ***" % (t / 60))
    return t


if __name__ == "__main__":
    notebook8()
    # notebook10()
    # notebook11()
    # notebook14()
