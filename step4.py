# TDGC analysis, functions for notebooks need sub-tasking
# corresponding to Chris's notebook 4

import time
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
from typing import Union

import warnings
warnings.filterwarnings("ignore")

import NotebookAnalysisUtilities as NAU
import DataUtilities3.MPL_rella as RMPL
import DataUtilities3.RellaColor as RC
RMPL.SetTheme('EIGHT')
KOL = RC.SeventiesFunk(brtValue=0.25)

from spectral_toolkit.db_clients.mongo_spectral_library import get_mongo_spectral_library_client
from spectral_toolkit.model_function.mongodb import model_function
from dask.diagnostics import ProgressBar
from dask import delayed, compute
from enum import Enum
from pydantic import Field
from spectral_arithmetic.linear_least_squares import least_squares

# define compound hunters
import picarro_xarray.accessors.compound_search_accessor  # noqa
import picarro_xarray.accessors.zero_reference_accessor  # noqa
from picarro_xarray.data_models.data_types import (
    TimeseriesDataArray,
    SpectralTimeseriesDataArray,
    CompactDataSet,
    ComboDataSet,
)
from picarro_xarray.data_models.constants import (
    TIME_DIM,
    MODE_DIM,
    FITTER_VARNAME,
    NU_VARS,
    NPOINTS_KEY,
)

from constants import SPLIT_TASK


class CorrelationType(str, Enum):
    SHORTTERM = "short_term"
    LONGTERM = "long_term"

    def slope_function(self):
        match self:
            case CorrelationType.SHORTTERM:
                return _slope_linear_trend
            case CorrelationType.LONGTERM:
                return _slope_lsq_fit_2D
            case _:
                raise ValueError(f"Unknown correlation type {self}")

def _lsq_fit(x: np.ndarray, y: np.ndarray, t: np.ndarray) -> float:
    """
    Using least_squares
    """
    Amatrix = np.vstack([np.ones_like(x), x, t]).T
    fit, _ = least_squares(Amatrix, y)
    return fit[1]


def _linear_trend(x: np.ndarray, y: np.ndarray) -> float:
    """
    Using numpy.polyfit to get a linear trend
    """
    trend = np.polyfit(x, y, 1)[0]
    return trend


def _detrend_dim(
    da: xr.DataArray,
    dim: str,
    deg: Field(default=1, ge=1),
) -> xr.DataArray:
    """
    detrend xarray DataArray along a single dimension
    """
    if dim not in da.dims:
        raise ValueError(f"Dimension {dim} not found in xarray DataArray")
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit


def _slope_linear_trend(
    da_spec: SpectralTimeseriesDataArray,
    da_timeseries: TimeseriesDataArray,
    detrend: bool = False,
) -> xr.DataArray:
    """
    Using numpy.polyfit and xr.apply_ufunc
    Can be optimized with dask
    """
    if detrend:
        da_spec = _detrend_dim(da_spec, dim=TIME_DIM, deg=1)
        da_timeseries = _detrend_dim(da_timeseries, dim=TIME_DIM, deg=1)
    out = xr.apply_ufunc(
        _linear_trend,
        da_timeseries,
        da_spec.dropna(dim=MODE_DIM),
        vectorize=True,
        input_core_dims=[[TIME_DIM], [TIME_DIM]],
    )
    return out


def _slope_lsq_fit_2D(
    da_spec: SpectralTimeseriesDataArray, 
    da_timeseries: TimeseriesDataArray
) -> xr.DataArray:
    """
    Using least_squares from spectral_arithmetic and xr.apply_ufunc
    Can be optimized with dask
    """
    times = da_timeseries[TIME_DIM]
    tmax = times.max()
    tmin = times.min()
    time_nondim = (times - tmin).astype(float) / (tmax - tmin).astype(float)
    da_time_itself = xr.DataArray(
        time_nondim, coords={TIME_DIM: times.values}, dims=[TIME_DIM]
    )
    out = xr.apply_ufunc(
        _lsq_fit,
        da_timeseries,
        da_spec.dropna(dim=MODE_DIM),
        da_time_itself,
        vectorize=True,
        input_core_dims=[[TIME_DIM], [TIME_DIM], [TIME_DIM]],
    )
    return out


def compute_correlation_spectrum(
    ds: Union[ComboDataSet, CompactDataSet],
    corr_key: str = 'partial_fit_integral',
    correlation_type: CorrelationType = CorrelationType.SHORTTERM,
) -> xr.Dataset:
    """
    Compute correlation spectrum
    Inputs :
        ds : (xr.Dataset)
        corr_key : (str) fitter_var with which to compute correlation
        correlation_type : (CorrelationType) short_term or long_term
    Returns :
        corrspec : (xr.Dataset) correlation spectrum centered at the middle of the event
    """
    da_spec = ds.spectral_values.dropna(dim=TIME_DIM, how='all').load()
    da_ts = ds.fitter_values.sel(fitter_var=corr_key).dropna(dim=TIME_DIM, how='all').load()
    tmid = (ds[TIME_DIM].max().values - ds[TIME_DIM].min().values) / 2 + ds[TIME_DIM].min().values
    fitter_val = (da_ts.max(TIME_DIM) - da_ts.min(TIME_DIM)).values
    corrspec = CorrelationType(correlation_type).slope_function()(da_spec, da_ts)
    spectral_var_indices = {k: i for i, k in enumerate(corrspec.spectral_var.values)}
    # Handle special treatment for "nu","nu_prime" and "npoints"
    for k in NU_VARS:
        corrspec[spectral_var_indices[k]] = da_spec.sel(
            spectral_var=k, mode=corrspec.mode
        ).isel({TIME_DIM: 0})
    corrspec[spectral_var_indices[NPOINTS_KEY]] = ds.spectral_values.sel(
        spectral_var=NPOINTS_KEY, mode=corrspec.mode
    ).sum(TIME_DIM)
    corrspec = corrspec.expand_dims({TIME_DIM: [tmid]})
    da_fitter = xr.DataArray(
        [fitter_val],
        coords={FITTER_VARNAME: [corr_key]},
        dims=[FITTER_VARNAME],
    )
    da_fitter = da_fitter.expand_dims({TIME_DIM: [tmid]})
    out = xr.merge(
        [
            da_fitter.to_dataset(name="fitter_values"),
            corrspec.to_dataset(name="spectral_values"),
        ]
    )
    return out


# ####### Step04_ComputeCorrelationSpectraInBin
def notebook4_part(n):
    print("********** start step 4 part %s: ********** ComputeCorrelationSpectraInBin ***" % n)
    t0 = time.time()

    config = NAU.load_analysis_yaml()
    dirs = NAU.AnalysisPaths(config['paths'])

    EXPT_START = config['times']['expt_start']
    EXPT_END = config['times']['expt_end']

    FITTING = config['fitting']
    nu_min = FITTING['nu_min']
    nu_max = FITTING['nu_max']
    nominal_pressure = FITTING['nominal_pressure']
    DONT_FIT = FITTING['dont_fit']
    exclude_list = FITTING['exclude_list']
    use_exact_pressure = FITTING['use_exact_pressure']

    NUM_REFS_TO_AVE = config['valve_info']['num_reference_periods']

    chromatogram_duration = config['chromatogram_params']['chromatogram_duration']
    corr_width = config['correlation_spectra']['step_width_sec']
    corr_step = config['correlation_spectra']['step_size_sec']
    corr_na_thresh = config['correlation_spectra']['na_thresh']
    corr_key = config['correlation_spectra']['corr_key']
    corr_range_thresh = config['correlation_spectra']['corr_key_range_threshold']
    corr_hi = config['correlation_spectra']['hi_pct']
    corr_lo = config['correlation_spectra']['lo_pct']

    TIMES = np.arange(0, chromatogram_duration*60, corr_step)
    print (corr_range_thresh, corr_lo, corr_hi)
    
    # get model functions
    client = get_mongo_spectral_library_client()
    cids = client.EmpiricalSpectra.distinct("cid")

    @delayed
    def get_model_function(cid):
        try:
            return cid, model_function(
                    client, 
                    cid=cid,
                    nu_min=nu_min,
                    nu_max=nu_max,
                    nominal_pressure=nominal_pressure,
                    use_exact_pressure=use_exact_pressure
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
    df_time = pd.read_parquet(dirs.misc_results_folder / 'chromatogram_times.parquet')
    print(df_time)
    
    # perform absolute spectral zero referencing for the entire time series
    earliest_time = min([df_time['start'].min(), df_time['ref_start'].min()])
    latest_time = min([df_time['end'].max(), df_time['ref_end'].max()])

    run_ds = ds.sel(_datetime=slice(earliest_time, latest_time)) #gather all the data
    run_ds = run_ds.dropna(dim='mode', how='all', subset=['spectral_values']) #drop data with nans in spectral values

    # create a mask for reference data
    for index, row in df_time.iterrows():
        if not row['good']: continue
        this_mask = (run_ds._datetime >= row['ref_start']) & (run_ds._datetime < row['ref_end'])
        if index == 0:
            zero_mask = this_mask
        else:
            zero_mask |= this_mask

    grouper = xr.where(zero_mask, 0, 1) # define background by data before the start time
    zero_ref_ds = run_ds.zero_reference.zero_reference(
        transition_variable=grouper,
        reference_value=0,
        event_window_width = NUM_REFS_TO_AVE
    )

    # compute correlation spectra
    KOL.resetCycle()
    pfi = zero_ref_ds.sel(fitter_var='partial_fit_integral').fitter_values.values
    
    # cut to 2~4 parts: every n element
    A = TIMES
    TIMES = A[n-1::SPLIT_TASK]    
    print('this part, len of TIMES', len(TIMES))

    for peak_id, ctime in enumerate(TIMES[:]):
        center = pd.Timedelta(ctime, 'sec')
        left = center - pd.Timedelta(corr_width/2, 'sec')
        right = center + pd.Timedelta(corr_width/2, 'sec')

        # select data in this time range
        data_mask = None
        metadata = []
        for index, row in df_time.iterrows():
            this_mask = (
                zero_ref_ds._datetime >= pd.to_datetime(row['start']) + left
            ) & (
                zero_ref_ds._datetime < pd.to_datetime(row['start']) + right
            )
            this_pfi_mean = pfi[this_mask].mean() 
            info = {'index': index, 'start': row['start'], 'sample_points': this_mask.values.sum(), 'mean_pfi': this_pfi_mean}
            metadata.append(info)
            if data_mask is None:
                data_mask = this_mask
            else:
                data_mask |= this_mask

        these_pfi = pfi[data_mask]
        lo, hi = (np.percentile(these_pfi, q) for q in [1, 99])
        if hi - lo <= corr_range_thresh: continue

        print(center.total_seconds(), f' range: {lo:.2f} - {hi:.2f}')
        analysis_ds = zero_ref_ds.sel(_datetime=data_mask).compute()   
        analysis_ds = analysis_ds.dropna(dim='mode', subset=['spectral_values'], thresh=corr_na_thresh)
        analysis_ds = analysis_ds.expand_dims({'peak_id' : [peak_id]})

        dslist = [analysis_ds]
        ds_peaks = xr.concat(dslist, dim='peak_id')

        ds_corr = ds_peaks.groupby('peak_id').apply(
            compute_correlation_spectrum, 
            corr_key=corr_key, 
            correlation_type=CorrelationType.LONGTERM
        )

        timebin_dir = Path(dirs.correlation_spectra_folder / f"BIN_{int(center.total_seconds()):05d}_sec")
        timebin_dir.mkdir(exist_ok=True, parents=True)
        test = ds_corr.sel(peak_id=peak_id).dropna(TIME_DIM, how='all')
        test.compound_search.set_model_functions(model_functions)
        X = test.compound_search.X
        Y = test.compound_search.Y.to_frame()

        # scale correlation by range of pfi
        Y[0] *= (hi - lo)
        bad_nu = Y.isna().sum(axis=1) == 1
        if bad_nu.sum() > 0:
            print(bad_nu.sum())
        X.to_parquet(timebin_dir / "X.parquet")
        Y.to_parquet(timebin_dir / "Y.parquet")

        A,F= RMPL.Maker()
        A.scatter(Y.index, Y.values, c=KOL[1], s=15)
        RMPL.setLabels(A, RMPL.wavenumbers, 'scaled abs. [ppb/cm]', f'{timebin_dir.parent.name}: {timebin_dir.name} [{lo:.1f} to {hi:.1f}]', fontsize=12)
        this_saver = RMPL.SaveFigure(DIR=timebin_dir).savefig
        this_saver(F, f'Fig_{timebin_dir.name}.png', dpi=150, closeMe=True)

    t = int(time.time() - t0)
    print("\n****************************************** ")
    print("**  Step  4 part %s done, took %.2f min ***" % (n, t/60))
    return t


if __name__ == "__main__":
    notebook4_part(2)




