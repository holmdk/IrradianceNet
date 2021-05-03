import matplotlib.pyplot as plt
import torch
import torchvision

def create_video(pred_Y, gt_Y, i, model_name):
    preds = pred_Y.cpu()[0]
    y_plot = gt_Y.cpu()[0]

    difference_plot = (torch.pow(pred_Y[0] - gt_Y[0], 2)).detach().cpu()

    # concat all images
    final_image = torch.cat([preds, y_plot, difference_plot], dim=0)

    # make them into a single grid image file
    grid = torchvision.utils.make_grid(final_image, nrow=2)

    filename = './results/{}/batch_{}.png'.format(model_name, str(i).zfill(4))
    torchvision.utils.save_image(grid, filename)
    plt.close()