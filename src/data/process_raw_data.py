import numpy as np
import pandas as pd
import xarray as xr
import torch
from src.data.utils.helper_functions import mask_latlon

# Load data
def get_data(config, variable):
    """
    Retrieve raw netcdf dataset of effective cloud albedo
    """
    data = xr.open_mfdataset(config['root_dir'] + variable + '*')[variable]
    # data = xr.open_mfdataset('C:/Users/Holm/Documents/IrradianceNet/data/SARAH/*')['CAL']
    return data


def mask_data(data):
    """
    Mask xarray data to specific area
    """
    return mask_latlon(data)


def interpolate_to_latslons(data, config):
    """
    Interpolate to specific lats lons given by interpolation factor.
    Set interpolation_factor=1 for full resolution, 4 for the low resolution dataset
    """
    if config['interpolation_factor'] > 1:
        lats = data.lat[::config['interpolation_factor']]
        lons = data.lon[::config['interpolation_factor']]
        data = data.interp(lat=lats, lon=lons, method='linear')
    return data

def interpolate_neighboring_nans(data):
    """
    Interpolate neighboring missing values
    """
    data = data.interpolate_na(dim='time', limit=1)
    data = data.interpolate_na(dim='lat', limit=1)
    data = data.interpolate_na(dim='lon', limit=1)
    return data


def remove_nans(data, config):
    """
    Remove NaNs according to percentage NaNs allowed for a particular timestamp that were not removed in previous function

    This mainly removes observations during the night when there is no solar irradiance.
    """
    # Remove nans
    summed_hours = data.groupby('time').count(dim=xr.ALL_DIMS)
    image_size = data.shape[1] * data.shape[2]
    n_values_needed = image_size - int(config['nans_allowed_percentage'] * image_size)
    return data.where(summed_hours > n_values_needed, drop=True)

def get_possible_starts(data, config):
    """
    Ensure that our past and future observations can happen in sequence"
    """
    difference_range = np.diff(data.time)
    frames_total = config['n_past_steps'] + config['n_future_steps']

    counted = np.zeros(difference_range.shape)
    for idx, time in enumerate(difference_range):
        if idx != counted.shape[0] - 1:
            if time == np.timedelta64(1800000000000, 'ns'):
                counted[idx + 1] = 1

    cum_sum = counted.copy()

    for idx, time in enumerate(counted):
        if idx > 0:
            if counted[idx] > 0:
                cum_sum[idx] = cum_sum[idx - 1] + cum_sum[idx]

    possible_indices = np.array(np.where(cum_sum >= (frames_total - 1))).ravel()  # 1 since it is index

    # we use the beginning of the sequence as index
    possible_starts = possible_indices - (frames_total - 1)
    possible_starts = possible_starts.astype('int')

    possible_starts.sort()
    return possible_starts


def transform(video, nan_to_num = True):
    """
    Transform xarray to torch.tensor
    """
    video_clip = np.array(video)
    if nan_to_num:
        video_clip = np.nan_to_num(video_clip)
    video_clip = torch.stack([torch.Tensor(i) for i in video_clip])
    return video_clip


def convert_from_CAL_to_k(data):
    """
    Transforms from CAL to k using k = (1 - CAL)
    """
    return 1 - data

def save(data, possible_starts, config):
    """
    Save cloud albedo as video in torch.tensor, possible starts in numpy array and timestamps in pandas dataframe
    """
    video_clip = transform(data)
    timestamps = data.time.values

    print('Saving torch data...')
    torch.save(video_clip,
               config['out_dir'] + config['nc_out_filename'].split('.')[0] + '.pt')

    print('Saving possible starts')
    with open(config['out_dir'] + config['nc_out_filename'].split('.')[0] + '_possible_starts' + '.npy',
              'wb') as f:
        np.save(f, possible_starts)

    print('Saving timestamps')
    timestamps = pd.DataFrame(data.time.values,
                                   columns=['StartTimeUTC'])  # [self.possible_starts]
    timestamps.to_csv(config['out_dir'] + config['nc_out_filename'].split('.')[0] + '_timestamps' + '.csv', index=False)


def process_data(config):
    """
    Process raw SARAH 2.1 data
    """
    # Pipeline
    cloud_albedo = get_data(config, 'CAL')
    cloud_albedo = mask_data(cloud_albedo)
    cloud_albedo = cloud_albedo.load()  # We do not load before masking, as doing so would require more memory
    cloud_albedo = interpolate_to_latslons(cloud_albedo, config)
    cloud_albedo = interpolate_neighboring_nans(cloud_albedo)
    cloud_albedo = remove_nans(cloud_albedo, config)
    cloud_albedo = convert_from_CAL_to_k(cloud_albedo)
    possible_starts = get_possible_starts(cloud_albedo, config)
    print('Saving CAL data...')
    save(cloud_albedo, possible_starts, config)

    if config['process_SIS']:
        sis_data = get_data(config, 'SIS')
        sis_data = mask_data(sis_data)
        sis_data = sis_data.load()
        print('Saving SIS data...')
        sis_data.to_netcdf(config['out_dir'] + config['nc_out_filename'].split('.')[0].replace('CAL', 'SIS') + '.nc')



if __name__ == '__main__':
    config = {
        'root_dir': '../../data/SARAH/',
        'out_dir': '../../data/',
        'nc_out_filename': 'CAL_2016_05.nc',
        'nans_allowed_percentage': 0.05,
        'n_past_steps': 4,
        'n_future_steps': 2,
        'interpolation_factor': 1,
        'process_SIS': True                # NOTE THIS IS OPTIONAL
    }

    print('Starting processing of SARAH 2.1 dataset...')
    try:
        process_data(config)
        print('Finished processing of SARAH 2.1 dataset...')
    except Exception as e:
        print('Could not process SARAH dataset due to {}, exiting script...'.format(e))
