import sys
import numpy as np
import torch
import cv2

def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]

    img = img.astype(np.float32)
    flow = flow.astype(np.float32)

    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR) # INTER_LINEAR
    return res

def create_flow(X, flow_model, params=None):
    assert flow_model in ('farneback', 'tvl1')

    B, H, W = X.shape[0], X.shape[3], X.shape[3]

    full_flow = np.zeros((B, H, W, 2))

    for b in range(B):

        # define frames to be used for flow calculation
        imgs_X = X[b].squeeze()
        im1 = imgs_X[-2].numpy()
        im2 = imgs_X[-1].numpy()

        # normalzie images
        norm_im1 = cv2.normalize(im1, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        norm_im2 = cv2.normalize(im2, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # compute dense optical flow
        if flow_model == 'farneback':
            flow = cv2.calcOpticalFlowFarneback(prev=norm_im1,
                                                next=norm_im2,
                                                flow=None,
                                                pyr_scale=0.5,
                                                levels=5,
                                                winsize=5,
                                                iterations=3,
                                                poly_n=3,
                                                poly_sigma=1.1,
                                                flags=0)
        elif flow_model == 'tvl1':
            optical_flow = cv2.optflow.DualTVL1OpticalFlow_create(
                tau=params['tau'], theta=params['theta'], nscales=params['n_scales'],
                warps=params['warps'], epsilon=params['epsilon'], innnerIterations=params['innnerIterations'],
                outerIterations=params['outerIterations'], scaleStep=params['scaleStep'], gamma=params['gamma'],
                medianFiltering=params['medianFiltering']
            )
            flow = optical_flow.calc(im1.astype(np.float32), im2.astype(np.float32), flow=None)
        else:
            sys.exit('Wrong flow model')


        full_flow[b] = flow

    return full_flow


def optflow_predict(X, flow_model='farneback', future=4, params=None):
    B, _, C, H, W = X.size()

    pred = torch.zeros((B, future, C, H, W))

    flow = create_flow(X, flow_model=flow_model, params=params) # flow_model)

    input = X[:, -1, :, :, :].squeeze().detach().numpy()

    for b in range(B):
        input_batch = input[b]
        for t in range(future):
            input_batch = warp_flow(input_batch, flow[b])
            pred[b, t, 0] = torch.tensor(input_batch)

    return pred