import torch
import numpy as np
import pandas as pd

from src.models.optical_flow_functions import optflow_predict
from src.config.config import CONFIG

from src.visualization.create_video import create_video

from src.data.IrradianceConverter import IrradianceConverter
from src.data.utils.helper_functions import convert_to_full_res, interpolate_borders


def predict_hres(data_loader, irradiance_converter, CONFIG):
    if CONFIG['pretrained_path'] is not None:
        if 'ckpt' in CONFIG['pretrained_path']:
            model_weights = torch.load(CONFIG['pretrained_path'])['state_dict']
        else:
            model_weights = torch.load(CONFIG['pretrained_path'])
        model_weights = {k.replace('model.', ''): v for k, v in model_weights.items()}
        CONFIG['model_arch'].load_state_dict(state_dict=model_weights)

    mae = []
    rmse = []

    mae_sis = []
    rmse_sis = []

    total_batches = len(data_loader)

    with torch.no_grad():

        for i, batch in enumerate(data_loader):
            print('\nProcessing batch {} out of {}'.format(i, total_batches))

            x, y, times = batch

            x = x.squeeze(2)
            y = y.squeeze()

            ts = times.numpy().squeeze()
            ts = pd.DataFrame(([pd.to_datetime(ts[x]).values for x in range(ts.shape[0])]))

            target_times = ts.iloc[:, - CONFIG['n_steps_ahead']:]

            if CONFIG['only_last_two_timesteps']:
                full_output_img = torch.zeros_like(x[:, -2:, :, 0])
                full_pred_img = torch.zeros_like(x[:, -2:, :, 0])
                target_times = target_times.iloc[:, -2:]
            else:
                full_output_img = torch.zeros_like(x)
                full_pred_img = torch.zeros_like(x)

            if CONFIG['patch_based']:
                img_size = (x.shape[2] * x.shape[4]) // 4
                patch_dim = img_size // 128

                for patch in range(x.shape[2]):
                    x_patch = x[:, :, patch]
                    x_patch = x_patch.permute(0, 1, 3, 4, 2)

                    y_hat = CONFIG['model_arch'].forward(x_patch.cuda()).squeeze()

                    if CONFIG['only_last_two_timesteps']:
                        y_hat = y_hat[:, -2:]
                        y = y[:, -2:]

                    full_pred_img[:, :, patch] = y_hat
                    full_output_img[:, :, patch] = y[:, :, patch]

                    if CONFIG['interpolate_borders']:
                        y_pred_plot = interpolate_borders(full_pred_img, patch_dim, 128, double=True)

                pred_Y = convert_to_full_res(full_pred_img, img_size, patch_dim, y.shape)
                gt_Y = convert_to_full_res(full_output_img, img_size, patch_dim, y.shape)
            else:
                if CONFIG['model_arch'] == 'opt_flow':
                    y_hat = optflow_predict(X=x[:, -2:].unsqueeze(2),
                                             flow_model=CONFIG['flow_model'],
                                             future=CONFIG['n_steps_ahead'],
                                             params=CONFIG['params'])  # tvl1
                    # plt.imshow(x[0, -1, :, :].numpy())
                    # plt.imshow(y_hat[0, 1, 0, :, :].numpy())
                else:
                    y_hat = CONFIG['model_arch'].forward(x.unsqueeze(4).cuda().float()).squeeze().detach().cpu().unsqueeze(2)

                if CONFIG['only_last_two_timesteps']:
                    y_hat = y_hat[:, -2:]
                    y = y[:, -2:]

                pred_Y = y_hat
                gt_Y = y.detach().cpu().unsqueeze(2)

            # CONVERT TO SIS
            pred_SIS = irradiance_converter.convert_k_to_SSI(pred_Y, target_times).squeeze()
            gt_SIS = irradiance_converter.return_sis(target_times)

            # Performance
            ## Albedo-related
            mae.append(torch.mean(abs(pred_Y - gt_Y)).item())
            rmse.append(torch.sqrt(torch.mean(torch.pow(pred_Y - gt_Y, 2))).item())

            ## Irradiance-related
            for batch in range(target_times.shape[1]):
                mae_sis.append(np.nanmean(abs(pred_SIS[:, batch].numpy() - gt_SIS[batch].values)))
                rmse_sis.append(np.sqrt(np.nanmean(np.power(pred_SIS[:, batch].numpy() - gt_SIS[batch].values, 2))))

            # OR HERE, BUT VIDEO LOOKS STRANGE!!
            if CONFIG['save_images']:
                create_video(pred_Y, gt_Y, i, model_name)

    # Remove Infs   # CHECK FOR MAXIMUM VALS
    mae = np.array(mae)
    mae = mae[~np.isinf(mae)]

    rmse = np.array(rmse)
    rmse = rmse[~np.isinf(rmse)]

    mae_sis = np.array(mae_sis)
    mae_sis = mae_sis[~np.isinf(mae_sis)]

    rmse_sis = np.array(rmse_sis)
    rmse_sis = rmse_sis[~np.isinf(rmse_sis)]

    return {'k_mae': np.nanmean(mae),
            'k_rmse': np.nanmean(rmse),
            'sis_mae': np.nanmean(mae_sis),
            'sis_rmse': np.nanmean(rmse_sis)}


if __name__ == '__main__':
    path = "../data/"

    model_name = 'low_res'

    " Test functionality "
    test_set = CONFIG[model_name]['dataset']

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=8,  # needs to be one for external dataloader to work
        num_workers=0,
        shuffle=False,
        pin_memory=False
    )

    irradiance_converter = IrradianceConverter(path, resolution='low' if model_name in ['low_res'] else 'high_res')  # 'opt_flow'

    # results = predict_hres(test_loader, irradiance_converter, CONFIG['high_res'])
    results = predict_hres(test_loader, irradiance_converter, CONFIG[model_name])
    print(results)

    # HIGH_RES
    # {'k_mae': 0.12004301851479018, 'k_rmse': 0.1825383138365862, 'sis_mae': 50.41529016812339, 'sis_rmse': 70.59029430499895}

    # OPTFLOW_HIGH_RES
    # {'k_mae': 0.24039468338819053, 'k_rmse': 0.33058442126929277, 'sis_mae': 105.9204860144326, 'sis_rmse': 153.92797622929677}


    # LOW-RES MODEL
    # {'k_mae': 0.24510830475319106, 'k_rmse': 0.32695535385346064, 'sis_mae': 112.02343463198096,
    #  'sis_rmse': 158.30714587677997}


    # {'k_mae': 0.24143351744533464, 'k_rmse': 0.33113294353102074, 'sis_mae': 107.73061761958793,
    # 'sis_rmse': 156.56728592570312}


    #TODO: TRY INTERPOLATION ALONG BORDERS!



