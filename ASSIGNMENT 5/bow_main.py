import numpy as np
import cv2
import glob
import os
from sklearn.cluster import KMeans
from tqdm import tqdm

import math
def findnn(D1, D2):
    """
    :param D1: NxD matrix containing N feature vectors of dim. D
    :param D2: MxD matrix containing M feature vectors of dim. D
    :return:
        Idx: N-dim. vector containing for each feature vector in D1 the index of the closest feature vector in D2.
        Dist: N-dim. vector containing for each feature vector in D1 the distance to the closest feature vector in D2
    """
    N = D1.shape[0]
    M = D2.shape[0]  # [k]

    # Find for each feature vector in D1 the nearest neighbor in D2
    Idx, Dist = [], []
    for i in range(N):
        minidx = 0
        mindist = np.linalg.norm(D1[i, :] - D2[0, :])
        for j in range(1, M):
            d = np.linalg.norm(D1[i, :] - D2[j, :])

            if d < mindist:
                mindist = d
                minidx = j
        Idx.append(minidx)
        Dist.append(mindist)
    return Idx, Dist


def grid_points(img, nPointsX, nPointsY, border):
    """
    :param img: input gray img, numpy array, [h, w]
    :param nPointsX: number of grids in x dimension
    :param nPointsY: number of grids in y dimension
    :param border: leave border pixels in each image dimension
    :return: vPoints: 2D grid point coordinates, numpy array, [nPointsX*nPointsY, 2]
    """

    w = img.shape[0]
    h = img.shape[1]
    vPoints = np.array([None,None])  # numpy array, [nPointsX*nPointsY, 2]

    grid_point_x = np.round(np.linspace(border,w-border, num = nPointsX,endpoint = True))
    grid_point_y = np.round(np.linspace(border,h-border, num = nPointsY,endpoint = True))

    for x_point in grid_point_x:
        for y_point in grid_point_y:
            new_point = np.array([int(x_point),int(y_point)])
            vPoints = np.vstack((vPoints,new_point))

    vPoints = np.delete(vPoints, 0, axis = 0)

    return vPoints


def descriptors_hog(img, vPoints, cellWidth, cellHeight):
    nBins = 8
    w = cellWidth
    h = cellHeight

    grad_x = cv2.Sobel(img, cv2.CV_16S, dx=1, dy=0, ksize=1)
    grad_y = cv2.Sobel(img, cv2.CV_16S, dx=0, dy=1, ksize=1)

    nBins = 8
    w = cellWidth
    h = cellHeight

    grad_x = cv2.Sobel(img, cv2.CV_16S, dx=1, dy=0, ksize=1)
    grad_y = cv2.Sobel(img, cv2.CV_16S, dx=0, dy=1, ksize=1)

    descriptors = np.zeros(128)  # list of descriptors for the current image, each entry is one 128-d vector for a grid point

    #Looping through the array of grid-points
    for i in range (len(vPoints)):
        point_i = vPoints[i]
        point_i_x = point_i[0]
        point_i_y = point_i[1]

        x_factors = [-2, -2, -2, -2, -1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1]
        y_factors = [-2, -1, 0, 1, -2, -1, 0, 1, -2, -1, 0,  1, -2, -1, 0, 1]
        cell_descriptor = []
        for j in range(16):
            row_cell = np.arange(point_i_x + x_factors[j] * cellWidth, point_i_x + (x_factors[j] + 1) * cellWidth)
            col_cell = np.arange(point_i_y + y_factors[j] * cellHeight, point_i_y + (y_factors[j] + 1) * cellHeight)

            cell_pixels = np.array([None,None])
            #Computing array of 2D-pixels present in each cell, storing them in cell_pixels

            for x_index in row_cell:
                for y_index in col_cell:
                    pixel = [x_index, y_index]
                    cell_pixels = np.vstack((cell_pixels, pixel))
            cell_pixels = np.delete(cell_pixels, 0, axis = 0)

            angles = np.array(None)

            #Computing histogram of gradients' orientation within each cell

            for pixel in cell_pixels:
                gradient_x = grad_x[int(pixel[0]),int(pixel[1])]
                gradient_y = grad_y[int(pixel[0]),int(pixel[1])]
                angle = np.arctan2(gradient_y, gradient_x)
                angles = np.append(angles, angle)
            angles = np.delete(angles,0,axis = 0)
            angles_histogram, _ = np.histogram(angles, nBins, (-math.pi,math.pi))
            angles_histogram = angles_histogram.tolist()
            cell_descriptor += angles_histogram
        descriptors = np.vstack((descriptors,cell_descriptor))
        # todo
    descriptors = np.delete(descriptors,0,axis = 0)

    #descriptors = np.asarray(descriptors) # [nPointsX*nPointsY, 128], descriptor for the current image (100 grid points)
    return descriptors



def create_codebook(nameDirPos, nameDirNeg, k, numiter):
    """
    :param nameDirPos: dir to positive training images
    :param nameDirNeg: dir to negative training images
    :param k: number of kmeans cluster centers
    :param numiter: maximum iteration numbers for kmeans clustering
    :return: vCenters: center of kmeans clusters, numpy array, [k, 128]
    """
    vImgNames = sorted(glob.glob(os.path.join(nameDirPos, '*.png')))
    vImgNames = vImgNames + sorted(glob.glob(os.path.join(nameDirNeg, '*.png')))

    nImgs = len(vImgNames)

    cellWidth = 4
    cellHeight = 4
    nPointsX = 10
    nPointsY = 10
    border = 8

    vFeatures = []  # list for all features of all images (each feature: 128-d, 16 histograms containing 8 bins)
    # Extract features for all image
    for i in tqdm(range(nImgs)):
        # print('processing image {} ...'.format(i+1))
        img = cv2.imread(vImgNames[i])  # [172, 208, 3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # [h, w]
        vPoints = grid_points(img,nPointsX,nPointsY,border)
        descriptors = descriptors_hog(img,vPoints,cellWidth,cellHeight)
        vFeatures.append(descriptors)
        # Collect local feature points for each image, and compute a descriptor for each local feature point
        # todo

    vFeatures = np.asarray(vFeatures)  # [n_imgs, n_vPoints, 128]
    vFeatures = vFeatures.reshape(-1, vFeatures.shape[-1])  # [n_imgs*n_vPoints, 128]
    print('number of extracted features: ', len(vFeatures))
    
    # Cluster the features using K-Means
    print('clustering ...')
    kmeans_res = KMeans(n_clusters=k, max_iter=numiter).fit(vFeatures)
    vCenters = kmeans_res.cluster_centers_  # [k, 128]

    return vCenters


def bow_histogram(vFeatures, vCenters):
    """
    :param vFeatures: MxD matrix containing M feature vectors of dim. D
    :param vCenters: NxD matrix containing N cluster centers of dim. D
    :return: histo: N-dim. numpy vector containing the resulting BoW activation histogram.
    """
    histo = np.zeros(len(vCenters))

    closest_centroids,_ = findnn(vFeatures, vCenters)  #MIGHT WORK AS WELL
    #histo,_ = np.histogram(closest_centroids, len(vCenters))

    for centroid in closest_centroids:
        histo[centroid] += 1

    return histo


def create_bow_histograms(nameDir, vCenters):
    """
    :param nameDir: dir of input images
    :param vCenters: kmeans cluster centers, [k, 128] (k is the number of cluster centers)
    :return: vBoW: matrix, [n_imgs, k]
    """
    vImgNames = sorted(glob.glob(os.path.join(nameDir, '*.png')))
    nImgs = len(vImgNames)

    cellWidth = 4
    cellHeight = 4
    nPointsX = 10
    nPointsY = 10
    border = 8

    # Extract features for all images in the given directory
    vBoW = []
    for i in tqdm(range(nImgs)):
        # print('processing image {} ...'.format(i + 1))
        img = cv2.imread(vImgNames[i])  # [172, 208, 3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # [h, w]
        vPoints = grid_points(img,nPointsX,nPointsY,border)
        vFeatures = descriptors_hog(img,vPoints,cellWidth,cellHeight)
        histo = bow_histogram(vFeatures, vCenters)
        vBoW.append(histo)
        # todo

    vBoW = np.asarray(vBoW)  # [n_imgs, k]

    return vBoW



def bow_recognition_nearest(histogram,vBoWPos,vBoWNeg):
    """
    :param histogram: bag-of-words histogram of a test image, [1, k]
    :param vBoWPos: bag-of-words histograms of positive training images, [n_imgs, k]
    :param vBoWNeg: bag-of-words histograms of negative training images, [n_imgs, k]
    :return: sLabel: predicted result of the test image, 0(without car)/1(with car)
    """

    DistPos, DistNeg = None, None

    # Find the nearest neighbor in the positive and negative sets and decide based on this neighbor
    # todo

    DistPos = np.min(np.linalg.norm(histogram - vBoWPos))
    DistNeg = np.min(np.linalg.norm(histogram - vBoWNeg))
    if (DistPos < DistNeg):
        sLabel = 1
    else:
        sLabel = 0
    return sLabel





if __name__ == '__main__':
    nameDirPos_train = 'data/data_bow/cars-training-pos'
    nameDirNeg_train = 'data/data_bow/cars-training-neg'
    nameDirPos_test = 'data/data_bow/cars-testing-pos'
    nameDirNeg_test = 'data/data_bow/cars-testing-neg'

    k = 10 # todo
    numiter = 500 # todo

    print('creating codebook ...')
    vCenters = create_codebook(nameDirPos_train, nameDirNeg_train, k, numiter)

    print('creating bow histograms (pos) ...')
    vBoWPos = create_bow_histograms(nameDirPos_train, vCenters)
    print('creating bow histograms (neg) ...')
    vBoWNeg = create_bow_histograms(nameDirNeg_train, vCenters)

    # test pos samples
    print('creating bow histograms for test set (pos) ...')
    vBoWPos_test = create_bow_histograms(nameDirPos_test, vCenters)  # [n_imgs, k]
    result_pos = 0
    print('testing pos samples ...')
    for i in range(vBoWPos_test.shape[0]):
        cur_label = bow_recognition_nearest(vBoWPos_test[i:(i+1)], vBoWPos, vBoWNeg)
        result_pos = result_pos + cur_label
    acc_pos = result_pos / vBoWPos_test.shape[0]
    print('test pos sample accuracy:', acc_pos)

    # test neg samples
    print('creating bow histograms for test set (neg) ...')
    vBoWNeg_test = create_bow_histograms(nameDirNeg_test, vCenters)  # [n_imgs, k]
    result_neg = 0
    print('testing neg samples ...')
    for i in range(vBoWNeg_test.shape[0]):
        cur_label = bow_recognition_nearest(vBoWNeg_test[i:(i + 1)], vBoWPos, vBoWNeg)
        result_neg = result_neg + cur_label
    acc_neg = 1 - result_neg / vBoWNeg_test.shape[0]
    print('test neg sample accuracy:', acc_neg)
