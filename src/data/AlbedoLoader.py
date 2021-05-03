"""
Data loading script for the effective cloud albedo dataset
"""
# Author: Andreas Holm Nielsen <ahn@eng.au.dk>
from numpy.core.multiarray import ndarray
from torch.utils.data import Dataset

import xarray as xr
import torch
import numpy as np
import pandas as pd


class AlbedoDataset(Dataset):
    """ Effective Cloud Albedo Dataset."""

    def __init__(self, root_dir: str, nc_filename: str, variable: str = 'k',
                 n_past_frames: int = 4, n_future_frames: int = 4,
                 return_timestamp: bool = True, include_metafeatures: bool = False) -> None:
        """

        Parameters
        ----------
        root_dir : str
            Absolute directory path where dataset is located

        nc_filename : str
            Name of netCDF files containing cloud albedo dataset

        variable : str
            The data variable of interest. Can be either 'k', 'SIS', 'SIS_clear' or 'all'.

        n_past_frames : int
            Number of past frames to be used in the encoding

        n_future_frames : int
            Number of future frames to predict

        return_timestamp : bool
            Should timesteps be returned in the form of torch.Tensor
        include_metafeatures : bool
            Include elevation map, currently with 0.00625 degree resolution

        """
        assert n_past_frames > 1
        assert variable in ['all', 'k', 'SIS', 'SIS_clear']

        self.root_dir = root_dir
        self.n_past_frames = n_past_frames
        self.n_future_frames = n_future_frames
        self.frames_total = self.n_past_frames + self.n_future_frames
        self.return_timestamp = return_timestamp
        self.include_metafeatures = include_metafeatures
        self.variable = variable

        self.video_clip = torch.load(root_dir + nc_filename.split('.')[0] + '_subset.pt')     #xr.open_dataset()

        # if variable == 'all':
        #     raise NotImplementedError('Loading all features is currently not supported')
        #     # self.cloud_albedo = xr.open_dataset(root_dir + nc_filename)
        # elif variable == 'k':
        #     if load_to_memory:
        #         self.cloud_albedo = xr.open_dataset(root_dir + nc_filename)[variable].load()
        #     else:
        #         self.cloud_albedo = xr.open_dataset(root_dir + nc_filename)[variable]
        #     self.cloud_albedo = self.cloud_albedo.sortby('time')
        # elif variable == 'SIS' or variable == 'SIS_clear':
        #     self.cloud_albedo = xr.open_dataset(root_dir + nc_filename).load()
        #     self.cloud_albedo = self.cloud_albedo.sortby('time')

        if return_timestamp:
            self.timestamps = pd.read_csv(root_dir + '/' + nc_filename.split('.')[0].replace('lres_', '') + '_subset_timestamps.csv')


        if include_metafeatures:
            # Time features
            hours = pd.to_datetime(self.cloud_albedo.time.values).hour/24
            days = pd.to_datetime(self.cloud_albedo.time.values).day/31
            months = pd.to_datetime(self.cloud_albedo.time.values).month/12

            def time_copy_to_lat_lon(timeframe):
                timeframe = np.expand_dims(timeframe, [1, 2])
                timeframe = np.repeat(timeframe, 128, 1)
                timeframe = np.repeat(timeframe, 128, 2)
                final_time = xr.DataArray(timeframe, dims=self.cloud_albedo.dims, coords=self.cloud_albedo.coords)
                return final_time

            def lat_lon_copy_to_time(data):
                data = np.expand_dims(data, 0)
                data = np.repeat(data, self.cloud_albedo.time.shape[0], 0)
                data = xr.DataArray(data, dims=self.cloud_albedo.dims, coords=self.cloud_albedo.coords)
                return data

            hours = time_copy_to_lat_lon(hours)
            days = time_copy_to_lat_lon(days)
            months = time_copy_to_lat_lon(months)

            self.cloud_albedo['hour'] = hours
            self.cloud_albedo['day'] = days
            self.cloud_albedo['month'] = months


            # elevation
            elevation = xr.open_dataarray("/media/oldL/data/data_sets/topography/elevation_0.00625.nc").load()
            elevation = elevation.sel(lat=self.cloud_albedo.lat,
                                      lon=self.cloud_albedo.lon)
            elevation = elevation / elevation.max()
            elevation = lat_lon_copy_to_time(elevation)

            self.cloud_albedo['elevation'] = elevation

            # Latitude and longitude features
            lons = np.expand_dims(self.cloud_albedo['lon'].values, 1).repeat(128, 1)
            lons = (lons - lons.min()) / (lons.max() - lons.min())
            lats = np.expand_dims(self.cloud_albedo['lat'].values, 0).repeat(128, 0)
            lats = (lats - lats.min()) / (lats.max() - lats.min())

            lons = lat_lon_copy_to_time(lons)
            lats = lat_lon_copy_to_time(lats)

            self.cloud_albedo['lat_feature'] = lats
            self.cloud_albedo['lon_feature'] = lons

        self.possible_starts = range(self.video_clip.shape[0] - 16)

    def __len__(self) -> None:
        return len(self.possible_starts)

    # def __getitem__(self, idx: int) -> Tuple[Any, Any, Union[torch.Tensor, list]]:
    def __getitem__(self, idx: int) -> torch.Tensor():
        index = self.possible_starts[idx]
        idx_end =(index + self.n_past_frames + self.n_future_frames)

        video_clip = self.video_clip[index:idx_end]

        if self.return_timestamp:
            times = self.timestamps[index:idx_end]
            times = times.astype('datetime64[s]').values.astype('int64')
            times = torch.tensor(times, dtype=torch.int64)
        else:
            times = []

        if self.include_metafeatures:
            y = video_clip[self.n_past_frames:, 0]
            X = video_clip[:self.n_past_frames]
        else:
            y = video_clip[self.n_past_frames:]
            X = video_clip[:self.n_past_frames].unsqueeze(1)  # [0:self.n_past_frames, :, :]

        # preprocess values
        if self.variable == 'k':
            X[X.isinf()] = 0
            y[y.isinf()] = 0
            X[X.isnan()] = 0
            y[y.isnan()] = 0
            X[X > 1.2] = 1.2
            X[X < 0] = 0
            y[y < 0] = 0
            y[y > 1.2] = 1.2

        return X, y, times


import sys

if __name__ == '__main__':
    if sys.platform == 'win32':
        path = "D:/data/"
    else:
        path = '/data/CAL_SIS_historical/'

    filename = '128x128_CAL2015.nc'  # full_CAL_SIS_1_of_8.nc  full res

    train_set = AlbedoDataset(root_dir=path,
                              nc_filename=filename,
                              variable='k',
                              n_past_frames=4,
                              n_future_frames=8,
                              return_timestamp=True,
                              include_metafeatures=True)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=16,
        shuffle=False,
        # num_workers=1
    )

    train_iter = iter(train_loader)
    #
    # for i in range(30):
    #     x = train_iter.next()
    #     if x.shape[0] != 16:
    #         print(x.shape)
    #         break

    # x, y, times = train_iter.next()

    X_train, y_train, timestamp = train_iter.next()

    timestamp = timestamp.numpy().astype('datetime64[s]')

    # Maybe try this for all observations to ensure nothing weird happens!
    print(np.diff(timestamp[0]))

    print('Video loaded')

    y_train.min()

    # import matplotlib.pyplot as plt
    # plt.imshow(X_train[0, 0, 0, :, :].detach().numpy())
    # plt.show()
    # plt.imshow(y_train[0, 3, :, :].detach().numpy())
    # plt.show()
    #


