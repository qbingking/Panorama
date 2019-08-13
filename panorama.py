from featureMatch import *
from homography import *

import cv2
import numpy as np
import glob
import argparse


def outputLimits(homography, imSize):
    height, width = imSize
    fourCorners = np.array(
        [[0, 0, 1], [0, height - 1, 1], [width - 1, 0, 1], [width - 1, height - 1, 1]])
    xMin, xMax = float('inf'), -float('inf')
    yMin, yMax = float('inf'), -float('inf')
    for corner in fourCorners:
        _pt = np.dot(homography, corner)
        _pt[0] /= _pt[2]
        _pt[1] /= _pt[2]

        xMin = min(xMin, _pt[0])
        xMax = max(xMax, _pt[0])

        yMin = min(yMin, _pt[1])
        yMax = max(yMax, _pt[1])

    return (xMin, xMax), (yMin, yMax)

 # stitching 2 hình lại theo best homo
def stitchImages(imLeft, imRight, homography):
    # compute the width and height 
    xlim, ylim = outputLimits(homography, imRight.shape[:2])

    heightLeft, widthLeft = imLeft.shape[:2]
    heightRight, widthRight = imRight.shape[:2]

    xMin = min(xlim[0], xlim[1], 0)
    xMax = max(xlim[0], xlim[1], widthLeft, widthRight)

    yMin = min(ylim[0], ylim[1], 0)
    yMax = max(ylim[0], ylim[1], heightLeft, heightRight)

    width = int(round(xMax - xMin))
    height = int(round(yMax - yMin))

    # Map dồn theo bên phải
    translateMatrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    if xMin < 0:
        translateMatrix[0, 2] = abs(xMin)
    if yMin < 0:
        translateMatrix[1, 2] = abs(yMin)
    homography = np.dot(translateMatrix, homography)
    result = cv2.warpPerspective(
        imRight, homography, (width, height))

    #  Map dồn theo bên trái
    xLow, xHigh = 0, imLeft.shape[1]
    yLow, yHigh = 0, imLeft.shape[0]
    if xMin < 0:
        xLow += abs(xMin)
        xHigh += abs(xMin)
    if yMin < 0:
        yLow += abs(yMin)
        yHigh += abs(yMin)

    xLow, xHigh = int(xLow), int(xHigh)
    yLow, yHigh = int(yLow), int(yHigh)

    result[yLow:yHigh, xLow:xHigh, :] = np.maximum(
        imLeft[:, :, :], result[yLow:yHigh, xLow:xHigh, :])

    result = result.astype('uint8')
    return result

#trả về mảng hình ảnh
def readImages(path):
    imgs = [] 
    for imgPath in sorted(glob.glob(path + '*')):
        img = cv2.imread(imgPath)
        img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        imgs.append(img)
    return imgs


def panorama(imgs):
    origin = imgs[0] # lấy hình đầu tiên làm chuẩn
    # có 5 hình input
    for i in range(1, len(imgs)): # 
        #print("Chuyển 2 hình L-R --> Gray ")
        left = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY)
        right = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2GRAY)
        
        #print("Tìm Keypoint & Des cho L R")
        kps1, des1 = extractSIFTFeature(left)
        kps2, des2 = extractSIFTFeature(right)
        
        # lấy những cặp keypoint ( 1 cái của Left, 1 cái của Right), trả về mảng của các cặp keypoint
        matchedDescriptors = getMatchedDescriptors(des2, des1)
        
        # Lấy tọa độ có liên quan, Lấy (x,y) của mỗi keypoint, pt tọa độ của kp pt[x],pt[y]
        pairCoords = getRelevantPairCoord(kps2, kps1, matchedDescriptors) # ((132, 238), (222, 251)), ((73, 115), (166, 123)), ((108, 136), (200, 145))
        h, pairPts = getBestHomography( # h:matran homo(3x3), mảng chứa các tọa độ keypoint [(100,150),(75,75),...]
            kps2, kps1, matchedDescriptors, pairCoords)

        origin = stitchImages(origin, imgs[i], h)
    
    return origin


if __name__ == '__main__':
    path = './img/test3/' # Truyền đường dẫn test data
    imgs = readImages(path)
    panoramaIm = panorama(imgs)

    panoramaIm = cv2.resize(panoramaIm, (0, 0), fx=0.9, fy=0.9)
    cv2.imshow('panorama', panoramaIm)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
