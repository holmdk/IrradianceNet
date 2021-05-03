from src.data.AlbedoPatchLoader import AlbedoDataset as AlbedoPatchDataset
from src.data.AlbedoLoader import AlbedoDataset as AlbedoDataset
from src.models.ConvLSTM_large import ConvLSTM_patch
from src.models.ConvLSTM_small import ConvLSTM

# TODO: Add argparse functionality, which saves to JSON or similar -- simply dump the settings written here!

path = '../data/'

CONFIG = {
  'high_res':
      {
          'nf': 128,
          'lr': 1e-4,
          'beta1': 0.9,
          'beta2': 0.999,
          'eps': 1e-8,
          'weight_decay': 0,
          'model_hyperparams': {'nf': 128, 'in_chan': 1, 'input_size': (128, 128)},  # TRY CHANGE in_chan
          'gamma': 0.5,
          'model_arch': ConvLSTM_patch(seq_len=2, in_chan=1, image_size=128).cuda(),
          'pretrained_path': '../models/full_res/convlstm_1h_ahead_patch.ckpt',
          'n_steps_ahead': 2,
          'only_last_two_timesteps': True,
          'save_images': True,
          'patch_based': True,
          'interpolate_borders': True,
          'dataset': AlbedoPatchDataset(root_dir=path,
                       nc_filename='CAL_2016_05.nc',
                       # nc_filename='test.nc',
                       variable='k',
                       n_past_frames=4,
                       n_future_frames=2,
                       return_timestamp=True,
                       patch_size=128,
                       train=False)
      },
  'low_res':
      {
          'nf': 128,
          'lr': 1e-3,
          'beta1': 0.9,
          'beta2': 0.999,
          'eps': 1e-8,
          'weight_decay': 0,
          'model_hyperparams': {'nf': 128, 'in_chan': 1, 'input_size': (128, 128)},
          'gamma': 0.5,
          'model_arch': ConvLSTM(seq_len=2, in_chan=1).cuda(), # 7
          'pretrained_path': '../models/small_res/45.pth',
          'n_steps_ahead': 2,
          'only_last_two_timesteps': True,
          'save_images': False,
          'patch_based': False,
          'dataset': AlbedoDataset(root_dir=path,
                       nc_filename='CAL_2016_05.nc',
                       variable='k',
                       n_past_frames=4,
                       n_future_frames=2,
                       return_timestamp=True)
      },
  'opt_flow':
      {
          'model_arch': 'opt_flow',
          'flow_model': 'tvl1',
          'pretrained_path': None,
          'params': {
              "tau": 0.3,
              "lambda": 0.21,
              "theta": 0.5,
              "n_scales": 3,
              "warps": 5,
              "epsilon": 0.01,
              "innnerIterations": 10,
              "outerIterations": 2,
              "scaleStep": 0.5,
              "gamma": 0.1,
              "medianFiltering": 5
          },
          'n_steps_ahead': 2,
          'only_last_two_timesteps': True,
          'save_images': True,
          'patch_based': False,
          'dataset': AlbedoDataset(root_dir=path,
                       nc_filename='CAL_2016_05.nc',
                       variable='k',
                       n_past_frames=4,
                       n_future_frames=2,
                       return_timestamp=True)
      }
}