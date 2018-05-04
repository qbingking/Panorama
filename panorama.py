from featureMatch import *
from homography import *

import cv2
import numpy as np
import glob


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
    # get the width and height of panorama image
    xlim, ylim = outputLimits(homography, imRight.shape[:2])

    heightLeft, widthLeft = imLeft.shape[:2]
    heightRight, widthRight = imRight.shape[:2]

    xMin = min(xlim[0], xlim[1], 0)
    xMax = max(xlim[0], xlim[1], widthLeft, widthRight)

    yMin = min(ylim[0], ylim[1], 0)
    yMax = max(ylim[0], ylim[1], heightLeft, heightRight)

    width = int(round(xMax - xMin))
    height = int(round(yMax - yMin))

    # result = cv2.warpPerspective(
    #     imRight, homography, (imRight.shape[1] + imLeft.shape[1], imRight.shape[0]))

    # get all points's coordinate from right image
    originCoords = np.where(imRight != None)
    originCoords = np.array(
        zip(originCoords[1], originCoords[0]), dtype='float64')
    originCoords = originCoords.reshape(-1, 1, 2)

    # get transformed coordinates
    transformedCoords = cv2.perspectiveTransform(originCoords, homography)

    # translate all points to the bottom right direction
    if xMin < 0:
        transformedCoords[:, :, 0] += abs(xMin)
    if yMin < 0:
        transformedCoords[:, :, 1] += abs(yMin)

    # reshape to two-dimensional array
    originCoords = originCoords.reshape(-1, 2)
    transformedCoords = transformedCoords.reshape(-1, 2)

    # Map right image to panorama image
    result = np.zeros((height, width, 3))

    for old, new in zip(originCoords, transformedCoords):
        newX, newY = int(new[0]) - 1, int(new[1]) - 1
        oldX, oldY = int(old[0]) - 1, int(old[1]) - 1
        result[newY, newX, :] = imRight[oldY, oldX, :]

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
    # cv2.imshow('figure', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return result


def readImages(path):
    imgs = []
    for imgPath in sorted(glob.glob(path + '*.jpg')):
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
    path = './img/moutainCopy/'
    imgs = readImages(path)
    panoramaIm = panorama(imgs)

    panoramaIm = cv2.resize(panoramaIm, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow('panoram', panoramaIm)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
