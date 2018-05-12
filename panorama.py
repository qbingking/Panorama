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


def stitchImages(imLeft, imRight, homography):
    # compute the width and height of panorama image
    xlim, ylim = outputLimits(homography, imRight.shape[:2])

    heightLeft, widthLeft = imLeft.shape[:2]
    heightRight, widthRight = imRight.shape[:2]

    xMin = min(xlim[0], xlim[1], 0)
    xMax = max(xlim[0], xlim[1], widthLeft, widthRight)

    yMin = min(ylim[0], ylim[1], 0)
    yMax = max(ylim[0], ylim[1], heightLeft, heightRight)

    width = int(round(xMax - xMin))
    height = int(round(yMax - yMin))

    # Map right image to panorama image
    translateMatrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    if xMin < 0:
        translateMatrix[0, 2] = abs(xMin)
    if yMin < 0:
        translateMatrix[1, 2] = abs(yMin)
    homography = np.dot(translateMatrix, homography)
    result = cv2.warpPerspective(
        imRight, homography, (width, height))

    # Map left image to panorama image
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


def readImages(path):
    imgs = []
    for imgPath in sorted(glob.glob(path + '*')):
        img = cv2.imread(imgPath)
        img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        imgs.append(img)
    return imgs


def panorama(imgs):
    origin = imgs[0]
    for i in range(1, len(imgs)):
        left = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY)
        right = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2GRAY)

        kps1, des1 = extractSIFTFeature(left)
        kps2, des2 = extractSIFTFeature(right)

        matchedDescriptors = getMatchedDescriptors(des2, des1)

        pairCoords = getRelevantPairCoord(kps2, kps1, matchedDescriptors)

        h, pairPts = getBestHomography(
            kps2, kps1, matchedDescriptors, pairCoords)

        origin = stitchImages(origin, imgs[i], h)
    return origin


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,
                    help="Path to the directory that contains the images to be stitched")
    args = vars(ap.parse_args())

    path = args["dataset"]
    imgs = readImages(path)
    panoramaIm = panorama(imgs)

    # panoramaIm = cv2.resize(panoramaIm, (0, 0), fx=0.4, fy=0.4)
    cv2.imshow('panorama', panoramaIm)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
