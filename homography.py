from featureMatch import extractSIFTFeature
from featureMatch import getMatchedDescriptors
from featureMatch import concatImages

import numpy as np
import cv2
from numpy import linalg as LA


def getRelevantPairCoord(kpts1, kpts2, matchedDescriptors):
    pairCoords = [((int(kpts1[id1].pt[0]), int(kpts1[id1].pt[1])),
                   (int(kpts2[id2].pt[0]), int(kpts2[id2].pt[1])))
                  for _, (id1, id2) in matchedDescriptors]

    return pairCoords


def estimateHomographyMatrix(kpts1, kpts2, matchedDescriptors, pairCoords):
    matchedDesSize = len(matchedDescriptors)
    ptsPairs = [pairCoords[np.random.randint(
        matchedDesSize)] for _ in range(4)]

    A = []
    for pp in ptsPairs:
        A.append([pp[0][0], pp[0][1], 1, 0, 0, 0, -pp[1][0]
                  * pp[0][0], -pp[1][0]*pp[0][1], -pp[1][0]])
        A.append([0, 0, 0, pp[0][0], pp[0][1], 1, -pp[1][1]
                  * pp[0][0], -pp[1][1]*pp[0][1], -pp[1][1]])
    A = np.array(A)

    # Use svd to calculate vh
    _, _, vh = LA.svd(A, full_matrices=True)

    # Then, select the last singular vector of vh
    return vh[-1].reshape(3, 3)


def manhattanDistance(p1, p2):
    # print p1, p2
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def getBestHomography(kps1, kps2, matchedDescriptors, pairCoords, threshold=5, iteration=100):
    '''Get the best homography using RANSAC algorithm'''
    bestCntInliers = -float('inf')
    bestPairPts = []
    bestHomography = None
    for i in range(iteration):
        homography = estimateHomographyMatrix(
            kps1, kps2, matchedDescriptors, pairCoords)

        cntInliners = 0
        pairPts = []
        for pcoord in pairCoords:
            pt = np.array([pcoord[0][0], pcoord[0][1], 1])
            pt = pt.reshape(3, 1)
            _pt = np.dot(homography, pt)
            _pt = (int(_pt.item(0) / _pt.item(2)),
                   int(_pt.item(1) / _pt.item(2)))

            dist = manhattanDistance(_pt, pcoord[1])
            # print dist

            if dist <= threshold:
                cntInliners += 1
                pairPts.append((pcoord[0], _pt))

        if cntInliners > bestCntInliers:
            bestCntInliers = cntInliners
            bestHomography = homography
            bestPairPts = pairPts

    print bestCntInliers
    # print bestHomography
    # print bestPairPts
    return bestHomography, bestPairPts


def drawMatchedPts(image1, image2, pairPts):
    outputImage = concatImages(image1, image2)
    for pt1, pt2 in pairPts:
        p2 = (pt2[0] + image1.shape[1], pt2[1])
        cv2.line(outputImage, pt1, p2, (0, 255, 0))
    return outputImage


if __name__ == '__main__':
    img1 = cv2.imread('./img/uttower/uttower_right.jpg')
    img2 = cv2.imread('./img/uttower/uttower_left.jpg')

    img1 = cv2.resize(img1, (0, 0), fx=0.5, fy=0.5)
    img2 = cv2.resize(img2, (0, 0), fx=0.5, fy=0.5)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    kps1, des1 = extractSIFTFeature(gray1)
    kps2, des2 = extractSIFTFeature(gray2)

    matchedDescriptors = getMatchedDescriptors(des1, des2)

    pairCoords = getRelevantPairCoord(kps1, kps2, matchedDescriptors)

    # print estimateHomographyMatrix(kps1, kps2, matchedDescriptors, pairCoords)
    h, pairPts = getBestHomography(kps1, kps2, matchedDescriptors, pairCoords)

    output = drawMatchedPts(img1, img2, pairPts)

    cv2.imshow('output_ransac', output)
    cv2.imwrite('./Output/output_ransac.png', output)

    panorama = stitchImages(img1, img2, h)
    cv2.imshow('panorama', panorama)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
