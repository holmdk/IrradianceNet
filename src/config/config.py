from src.data.AlbedoPatchLoader import AlbedoDataset as AlbedoPatchDataset
from src.data.AlbedoLoader import AlbedoDataset as AlbedoDataset
from src.models.ConvLSTM_large import ConvLSTM_patch

def config_parser(args):
    assert args.model_name in ['convlstm', 'opt_flow'], 'Model name must be either "convlstm" or "opt_flow"'

    if args.model_name == 'convlstm':
        CONFIG = {
                'model_name': args.model_name,
                'nf': 128,
                'lr': 1e-4,
                'beta1': 0.9,
                'beta2': 0.999,
                'eps': 1e-8,
                'weight_decay': 0,
                'model_hyperparams': {'nf': 128, 'in_chan': 1, 'input_size': (128, 128)},
                'gamma': 0.5,
                'model_arch': ConvLSTM_patch(seq_len=args.n_future_frames, in_chan=args.in_channel,
                                             image_size=128).cuda(),
                'pretrained_path': './models/full_res/convlstm_1h_ahead_patch.ckpt',
                'n_steps_ahead': args.n_future_frames,
                'save_images': args.save_images,
                'patch_based': True,
                'interpolate_borders': args.interpolate_patch_borders,
                'dataset': AlbedoPatchDataset(root_dir=args.data_path,
                                              nc_filename=args.cal_filename + '.nc',
                                              variable='k',
                                              n_past_frames=args.n_past_frames,
                                              n_future_frames=args.n_future_frames,
                                              return_timestamp=True,
                                              patch_size=128,
                                              train=False)
            }
    else:
        CONFIG = {
          'model_name': args.model_name,
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
          'n_steps_ahead': args.n_future_frames,
          'save_images': args.save_images,
          'patch_based': False,
          'dataset': AlbedoDataset(root_dir=args.data_path,
                       nc_filename=args.cal_filename + '.nc',
                       variable='k',
                       n_past_frames=args.n_past_frames,
                       n_future_frames=args.n_future_frames,
                       return_timestamp=True)
      }
    return CONFIG