import time
import os
import random
import math
import torch
import numpy as np

# run `pip install scikit-image` to install skimage, if you haven't done so
from skimage import io, color
from skimage.transform import rescale

def distance(x, X):

    dist = np.sqrt(((x-X)**2).sum(1))

    return(dist)
    #raise NotImplementedError('distance function not implemented!')

def distance_batch(x, X):

    dist = np.sqrt(((x[:,None] - X[None,:]) ** 2).sum(2))

    return(dist)
    #raise NotImplementedError('distance_batch function not implemented!')

def gaussian(dist, bandwidth):

    weight = (np.exp(-(dist**2)/ (2 * (bandwidth**2)))) / (bandwidth * np.sqrt(2*math.pi))

    return(weight)
    #raise NotImplementedError('gaussian function not implemented!')

def update_point(weight, X):

    pointupdate = ((weight[:,None]*X).sum(0)) / (weight.sum())   #OPTIMAL TIME

    return(pointupdate)
    #raise NotImplementedError('update_point function not implemented!')

def update_point_batch(weight, X):

    batchupdate = ((weight[:,:,None] * X).sum(1)) / (weight.sum(1)[:,None])

    return(batchupdate)
    #raise NotImplementedError('update_point_batch function not implemented!')

def meanshift_step(X, bandwidth=2.5):
    X_ = X.clone()
    for i, x in enumerate(X):
        dist = distance(x, X)
        weight = gaussian(dist, bandwidth)
        X_[i] = update_point(weight, X)

    return X_

def meanshift_step_batch(X, bandwidth=2.5):
    X_ = X.clone()
    BATCH_SIZE = 49
    BATCH_NUMBER = 75

    for batch in range(BATCH_NUMBER):

        dist = distance_batch(X[(batch * BATCH_SIZE) : ((batch+1) * BATCH_SIZE)],X)
        weight = gaussian(dist, bandwidth)
        X_[(batch * BATCH_SIZE) : ((batch+1) * BATCH_SIZE)] = update_point_batch(weight,X)

    return X_
    #raise NotImplementedError('meanshift_step_batch function not implemented!')

def meanshift(X):
    X = X.clone()
    for _ in range(20):
        #X = meanshift_step(X)   # slow implementation
        X = meanshift_step_batch(X)   # fast implementation
    return X

scale = 0.25  # downscale the image to run faster

# Load image and convert it to CIELAB space
image = rescale(io.imread('cow.jpg'), scale, multichannel=True)
image_lab = color.rgb2lab(image)
shape = image_lab.shape # record image shape
image_lab = image_lab.reshape([-1, 3])  # flatten the image

# Run your mean-shift algorithm
t = time.time()
X = meanshift(torch.from_numpy(image_lab)).detach().cpu().numpy()
#X = meanshift(torch.from_numpy(data).cuda()).detach().cpu().numpy()  # you can use GPU if you have one
t = time.time() - t
print ('Elapsed time for mean-shift: {}'.format(t))

# Load label colors and draw labels as an image
colors = np.load('colors.npz')['colors']
colors[colors > 1.0] = 1
colors[colors < 0.0] = 0

centroids, labels = np.unique((X / 4).round(), return_inverse=True, axis=0)

result_image = colors[labels].reshape(shape)
result_image = rescale(result_image, 1 / scale, order=0, multichannel=True)     # resize result image to original resolution
result_image = (result_image * 255).astype(np.uint8)
io.imsave('result.png', result_image)
