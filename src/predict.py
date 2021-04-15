import torch
import pandas as pd

from models.ConvLSTM_large import ConvLSTM_patch
from models.ConvLSTM_small import ConvLSTM_small

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
            'model_arch': ConvLSTM_patch(seq_len=2, in_chan=1, image_size=128),
            'save_path': '/home/local/DAC/ahn/Documents/dcwis.torch_forecasting/checkpoints/convlstm_1h_ahead_patch.ckpt'
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
                'model_arch': ConvLSTM_small(seq_len=2, in_chan=7),
                'save_path': '/home/local/DAC/ahn/Documents/dcwis.torch_forecasting/checkpoints/convlstm_1h_ahead.ckpt'
            }
}






def predict_hres(data_loader):
    total_batches = len(data_loader)

    with torch.no_grad():

        for i, batch in enumerate(data_loader):
            print('\nProcessing batch {} out of {}'.format(i, total_batches))
            x, y, times = batch

            #TODO: Remove this below

            y[y.isinf()] = 0
            y[y.isnan()] = 0
            y[y < 0] = 0
            x = x.squeeze(2)
            y = y.squeeze()

            ts = times.numpy().squeeze()
            ts = pd.DataFrame(([pd.to_datetime(ts[x]).values for x in range(ts.shape[0])]))
            # ts = np.array([pd.to_datetime(ts[x]).values for x in range(ts.shape[0])])

            target_times = ts.iloc[:, - self.n_steps_ahead:]
            diff_time = 0


            print('Data loading took {} seconds'.format(round(time.time() - start_timer, 2)))
            start_timer = time.time()

            img_size = (x.shape[2] * x.shape[4]) // 4
            patch_dim = img_size // 128

            if self.only_last_two_timesteps:
                start_time = target_times.iloc[0, 0]
                full_output_img = torch.zeros_like(x[:, -2:, :, 0])
                full_pred_img = torch.zeros_like(x[:, -2:, :, 0])
                target_times = target_times.iloc[:, -2:]
                diff_time = (target_times.iloc[0, 0] - start_time).seconds
                diff_time = diff_time // 60
            else:
                full_output_img = torch.zeros_like(x)
                full_pred_img = torch.zeros_like(x)

            # TODO: WE ARE DOING PATCHES, SO WE NEED TO ITERATE OVER ALL PATCHES PER IMAGE AND THEN CONCATENATE AFTERWARDS!
            for patch in range(x.shape[2]):
                # break
                x_patch = x[:, :, patch]
                x_patch = x_patch.permute(0, 1, 3, 4, 2)
                y_hat = mdl.forward(x_patch.cuda(), softmax=False).squeeze()

                if self.only_last_two_timesteps:
                    y_hat = y_hat[:, -2:]
                    y = y[:, -2:]

                full_pred_img[:, :, patch] = y_hat
                full_output_img[:, :, patch] = y[:, :, patch]

            pred_Y = full_pred_img.view(y.shape[0], y.shape[1], patch_dim, patch_dim, 1, 128, 128)
            pred_Y = pred_Y.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
            pred_Y = pred_Y.view(y.shape[0], y.shape[1], 1, img_size, img_size)