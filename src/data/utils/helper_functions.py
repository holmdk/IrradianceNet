import numpy as np

def convert_to_full_res(img, img_size, patch_dim, shapes):
    img = img.view(shapes[0], shapes[1], patch_dim, patch_dim, 1, 128, 128)
    img = img.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
    img = img.view(shapes[0], shapes[1], 1, img_size, img_size)
    return img


def remove_duplicates(data):
    _, index = np.unique(data.time.values, return_index=True)
    return data.isel(time=index)

def interpolate_along_dim(y_pred, idx_start, idx_end, t, axis='rows'):
    if axis == 'rows':
        start_vals = y_pred[:, :, idx_start - 1:idx_end + 1][t][0][0]
        end_vals = y_pred[:, :, idx_start - 1:idx_end + 1][t][-1][0]
    else:
        start_vals = y_pred[:, :, idx_start - 1:idx_end + 1][t][:, 0]
        end_vals = y_pred[:, :, idx_start - 1:idx_end + 1][t][:, -1]

    diff_interpolate = (start_vals - end_vals) / 7
    diff_interpolate = diff_interpolate.unsqueeze(0)  # .repeat(6, 1)
    diff_interpolate = diff_interpolate.repeat(6, 1)

    vals = np.arange(1, 7)
    vals = np.expand_dims(vals, 0)
    vals = np.repeat(vals, diff_interpolate.shape[1], axis=0)

    interpol_values = diff_interpolate * vals.T
    interpol_values = start_vals.unsqueeze(0).repeat(6, 1) - interpol_values
    return interpol_values

def interpolate_borders(y_pred, patch_dim, patch_size, double=True):
    " INTERPOLATE BORDERS "
    # We mask the pixels we want to interpolate
    for patch_border in range(patch_dim - 1):
        idx_start = (patch_border + 1) * patch_size - 3
        idx_end = idx_start + 6

        for b in range(y_pred.shape[0]):
            y_pred[b][:, idx_start:idx_end] = np.nan
            y_pred[b][:, :, idx_start:idx_end] = np.nan
            for t in range(y_pred.shape[1]):
                if double:
                    for _ in range(2):  # repeat interpolation to avoid nans
                        interpol_values_rows = interpolate_along_dim(y_pred[b], idx_start, idx_end, t, axis='rows')
                        interpol_values_cols = interpolate_along_dim(y_pred[b], idx_start, idx_end, t, axis='cols')

                        y_pred[b, t, :, idx_start:idx_end] = interpol_values_rows
                        y_pred[b, t, :, idx_start:idx_end] = interpol_values_cols.T
                else:
                    interpol_values_rows = interpolate_along_dim(y_pred[b], idx_start, idx_end, t, axis='rows')
                    interpol_values_cols = interpolate_along_dim(y_pred[b], idx_start, idx_end, t, axis='cols')

                    y_pred[b, t, idx_start:idx_end] = interpol_values_rows
                    y_pred[b, t, :, idx_start:idx_end] = interpol_values_cols.T

    return y_pred
