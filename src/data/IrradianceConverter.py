import xarray as xr
import torch
from src.data.utils.helper_functions import *

class IrradianceConverter:
    def __init__(self, path, sis_name='SIS_2016_05.nc', sis_clearsky_name='irradiance_2016_05.nc', resolution='high'):
        self.resolution = resolution

        self.clearsky = xr.open_dataset(path + sis_clearsky_name)  # t
        self.clearsky = remove_duplicates(self.clearsky)
        self.clearsky['SIS_clear'] = self.clearsky['SIS_clear'].clip(min=0, max=self.clearsky['SIS_clear'].quantile(0.99).item())  # we have a few outliers

        self.SIS = xr.open_dataset(path + sis_name)
        self.SIS = remove_duplicates(self.SIS)
        self.SIS['SIS'] = self.SIS['SIS'].clip(min=0, max=self.SIS['SIS'].quantile(0.99).item())  # we have a few outliers

        if resolution == 'low':
            self.coords = xr.open_dataset(path + 'lres_test_SIS.nc')  # t
            self.coords = {'lat': self.coords.lat, 'lon':self.coords.lon}

    def convert_k_to_SSI(self, data, target_times):
        sis_data = torch.zeros_like(data)

        for batch in range(target_times.shape[1]):
            sis_clear = self.clearsky.sel(time=target_times.iloc[:, batch].values)['SIS_clear']
            if sis_clear.shape[-1] != data.shape[-1]:
                sis_clear = sis_clear.sel(lat=self.coords['lat'], lon=self.coords['lon'])

            sis_data[:, batch, 0] = data[:, batch, 0] * sis_clear.values

        return sis_data

    def return_sis(self, target_times):
        sis_list = []
        for batch in range(target_times.shape[1]):
            sis_vals = self.SIS['SIS'].sel(time=target_times.iloc[:, batch].values)
            if self.resolution == 'low':
                sis_vals = sis_vals.sel(lat=self.coords['lat'], lon=self.coords['lon'])
            sis_list.append(sis_vals)

        return sis_list
