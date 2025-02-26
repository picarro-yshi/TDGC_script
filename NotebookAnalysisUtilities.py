import yamale 
from pathlib import Path
import pandas as pd
import xarray as xr

def load_analysis_yaml(yaml_name='./analysis_config.yaml'):
    config = yamale.make_data(yaml_name)[0][0]
    for key, value in config['times'].items():
        config['times'][key] = pd.to_datetime(value)
    return config

class AnalysisPaths():
    def __init__(self, path_config):
        self.path_config = path_config

    def subfolder(self, subdir_name, create_folder=False):
        this_path = self.exp_folder / self.path_config[subdir_name]
        if create_folder:
            this_path.mkdir(parents=True, exist_ok=True)
        return this_path
    
    @property
    def exp_folder(self):
        return Path(self.path_config['exp_folder'])

    @property
    def combo_folder(self):
        return self.subfolder('combo_logs')

    @property
    def TDGC_logs(self):
        return self.subfolder('TDGC_logs')
        
    @property
    def zarr_path(self):
        return self.subfolder('zarr_name')

    @property
    def misc_results_folder(self):
        return self.subfolder('misc_results', create_folder=True)

    @property
    def average_spectra_folder(self):
        return self.subfolder('average_spectra', create_folder=True)

    @property
    def correlation_spectra_folder(self):
        return self.subfolder('correlation_spectra', create_folder=True)
    
    @property
    def refit_folder(self):
        return self.subfolder('refit', create_folder=True)

    def __repr__(self):
        txt_lst = [str(self.exp_folder)]
        for key in self.path_config.keys():
            if key == 'exp_folder': continue
            
            txt_lst.append(f'- {key}: {self.subfolder(key)}')
        txt = '\n'.join(txt_lst)
        return txt

def load_ds_from_zarr(zarr_path):
    ds = xr.open_zarr(zarr_path)
    ds = ds.drop_vars(["fitter_origin_file"])
    ds.close()
    return ds
