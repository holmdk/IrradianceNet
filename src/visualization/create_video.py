import matplotlib.pyplot as plt
import torch
import torchvision


def create_video(pred_Y, gt_Y, i, model_name):
    " PLOT IMAGE NOW "

    # plt.imshow(x_full[0, 0, 0].detach().cpu().numpy())
    # plt.imshow(x_full[0, 1, 0].detach().cpu().numpy())
    # plt.imshow(x_full[0, 2, 0].detach().cpu().numpy())
    # plt.imshow(x_full[0, 3, 0].detach().cpu().numpy())
    #
    # preds = torch.cat([x_full.cpu(), pred_Y.cpu()], dim=1)[0]
    # y_plot = torch.cat([x_full.cpu(), gt_Y.cpu()], dim=1)[0]
    preds = pred_Y.cpu()[0]
    y_plot = gt_Y.cpu()[0]

    # error (l2 norm) plot between pred and ground truth
    # difference = (torch.pow(pred_Y[0] - gt_Y[0], 2)).detach().cpu()
    # zeros = torch.zeros(x_full[0].shape)
    # difference_plot = torch.cat([zeros.cpu().unsqueeze(0), difference.unsqueeze(0).cpu()], dim=1)[0]
    difference_plot = (torch.pow(pred_Y[0] - gt_Y[0], 2)).detach().cpu()

    # concat all images
    final_image = torch.cat([preds, y_plot, difference_plot], dim=0)

    # make them into a single grid image file
    grid = torchvision.utils.make_grid(final_image, nrow=2)

    filename = 'C:/Users/Holm/Documents/IrradianceNet/results/{}/batch_{}.png'.format(model_name, str(i).zfill(4))
    torchvision.utils.save_image(grid, filename)
    plt.close()