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
    """
    Full-resolution Effective Cloud Albedo Dataset with patch loading
    """

    def __init__(self, root_dir: str, nc_filename: str, variable: str = 'k',
                 n_past_frames: int = 4, n_future_frames: int = 4,
                 return_timestamp: bool = True,
                 patch_size: int = 32, train: bool = True) -> None:
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

        """
        assert n_past_frames > 1
        assert variable in ['k', 'SIS']

        self.root_dir = root_dir
        self.n_past_frames = n_past_frames
        self.n_future_frames = n_future_frames
        self.frames_total = self.n_past_frames + self.n_future_frames
        self.return_timestamp = return_timestamp
        self.variable = variable
        self.patch_size = patch_size
        self.train = train

        self.video_clip = torch.load(root_dir + nc_filename.split('.')[0] + '.pt')     #xr.open_dataset()  # '_subset.pt'

        if return_timestamp:
            self.timestamps = pd.read_csv(root_dir + '/' + nc_filename.split('.')[0] + '_timestamps.csv')

        self.possible_starts = np.load(root_dir + nc_filename.split('.')[0] + '_possible_starts.npy')

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


        X = video_clip[:self.n_past_frames]
        if self.train:
            y = video_clip[self.n_past_frames:, 0]
        else:
            y = video_clip[self.n_past_frames:, :, 0]


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

