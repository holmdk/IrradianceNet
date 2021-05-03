"""
Data loading script for the effective cloud albedo dataset
"""
# Author: Andreas Holm Nielsen <ahn@eng.au.dk>

from torch.utils.data import Dataset
import torch
import sys
import numpy as np
import pandas as pd


class AlbedoDataset(Dataset):
    """ Effective Cloud Albedo Dataset."""

    def __init__(self, root_dir: str, nc_filename: str, variable: str = 'k',
                 n_past_frames: int = 4, n_future_frames: int = 4,
                 return_timestamp: bool = True,
                 include_metafeatures: bool = False,
                 patch_size: int = 32, train: bool = True, use_cached_starts: bool = True) -> None:
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
        self.patch_size = patch_size
        self.train = train

        self.video_clip = torch.load(root_dir + nc_filename.split('.')[0] + '_subset.pt')     #xr.open_dataset()

        if return_timestamp:
            self.timestamps = pd.read_csv(root_dir + '/' + nc_filename.split('.')[0] + '_subset_timestamps.csv')

        if self.include_metafeatures:
            self.channels, self.height, self.width = self.video_clip.size()[1:]
        else:
            self.channels = 1
            self.height, self.width = self.video_clip.size()[1:]

        self.possible_starts = range(self.video_clip.shape[0]-16)

    def __len__(self) -> int:
        return len(self.possible_starts)

    # def __getitem__(self, idx: int) -> Tuple[Any, Any, Union[torch.Tensor, list]]:
    def __getitem__(self, idx: int) -> torch.Tensor():
        index = self.possible_starts[idx]
        idx_end = (index + self.n_past_frames + self.n_future_frames)
        video_clip = self.video_clip[index:idx_end]

        if self.train:
            # SELECT RANDOM PATCH HERE
            i = torch.randint(0, self.height-self.patch_size + 1, size=(1, )).item()
            j = torch.randint(0, self.width-self.patch_size + 1, size=(1, )).item()
            video_clip = video_clip[..., i : i + self.patch_size, j:j + self.patch_size]

        else:
            # At inference, we iterate over all patches to create our full image
            if not self.include_metafeatures:
                video_clip = video_clip.unsqueeze(1)

            kc, kh, kw = video_clip.shape[1], self.patch_size, self.patch_size
            dc, dh, dw = video_clip.shape[1], self.patch_size, self.patch_size
            video_clip = video_clip.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)

            video_clip = video_clip.contiguous().view(video_clip.size(0), -1, kc, kh, kw)

        if self.return_timestamp:
            times = self.timestamps[index:idx_end]
            times = times.astype('datetime64[s]').values.astype('int64')
            times = torch.tensor(times, dtype=torch.int64)
        else:
            times = []

        if self.include_metafeatures:
            X = video_clip[:self.n_past_frames]
            if self.train:
                y = video_clip[self.n_past_frames:, 0]
            else:
                y = video_clip[self.n_past_frames:, :, 0]
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


if __name__ == '__main__':
    if sys.platform == 'win32':
        path = "F:/Google Drive/Andreas/Post Study/Industrial/Code/IrradianceNet/"
    else:
        # path = '/data/CAL_SIS_historical/'
        path = '/media/oldL/data/data_sets/CMSAF/high_resolution/'

    filename = 'test.nc'  # 128x128_CAL2015

    " Train functionality "
    train_set = AlbedoDataset(root_dir=path,
                              nc_filename=filename,
                              variable='k',
                              n_past_frames=4,
                              n_future_frames=8,
                              return_timestamp=True,
                              include_metafeatures=False,
                              patch_size=128,
                              train=True)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=12,
        shuffle=False,
        # num_workers=1
    )

    train_iter = iter(train_loader)

    X_train, y_train, timestamp = train_iter.next()

    timestamp = pd.to_datetime(timestamp.numpy().ravel())

    " Test functionality "
    test_set = AlbedoDataset(root_dir=path,
                             nc_filename='test.nc',
                             variable='k',
                             n_past_frames=4,
                             n_future_frames=6,
                             return_timestamp=True,
                             include_metafeatures=False,
                             patch_size=128,
                             train=False)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=8,  # needs to be one for external dataloader to work
        num_workers=0,
        shuffle=False,
        pin_memory=False
    )

    test_iter = iter(test_loader)

    X_test, y_test, timestamp = test_iter.next()

    timestamp = pd.to_datetime(timestamp.numpy().ravel())

