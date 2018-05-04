from featureMatch import *
from homography import *

import cv2
import numpy as np
import glob


def stitchImages(imLeft, imRight, homography):
    result = cv2.warpPerspective(
        imRight, homography, (imRight.shape[1] + imLeft.shape[1], imRight.shape[0]))
    result[0:imLeft.shape[0], 0:imLeft.shape[1]] = np.maximum(
        imLeft, result[0:imLeft.shape[0], 0:imLeft.shape[1]])
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
    path = './img/moutain/'
    imgs = readImages(path)
    panoramaIm = panorama(imgs)

    cv2.imshow('panoram', panoramaIm)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
