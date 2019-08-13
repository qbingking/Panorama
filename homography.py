from featureMatch import extractSIFTFeature
from featureMatch import getMatchedDescriptors
from featureMatch import concatImages

import numpy as np
import cv2
from numpy import linalg as LA


def getRelevantPairCoord(kpts1, kpts2, matchedDescriptors):
    pairCoords = [((int(kpts1[id1].pt[0]), int(kpts1[id1].pt[1])), # Lấy (x,y) của mỗi keypoint, pt tọa độ của kp pt[x],pt[y]
                   (int(kpts2[id2].pt[0]), int(kpts2[id2].pt[1])))
                  for _, (id1, id2) in matchedDescriptors]
    # print("Pair",pairCoords)    ((132, 238), (222, 251)), ((73, 115), (166, 123)), ((108, 136), (200, 145))
    # die()                                  
    return pairCoords

#Tìm ma trận homography
def estimateHomographyMatrix(kpts1, kpts2, matchedDescriptors, pairCoords):
    matchedDesSize = len(matchedDescriptors)
    ptsPairs = [pairCoords[np.random.randint(
        matchedDesSize)] for _ in range(4)]

    A = []
    for pp in ptsPairs:#((132, 238), (222, 251))
        A.append([pp[0][0], pp[0][1], 1, 0, 0, 0, -pp[1][0] * pp[0][0], -pp[1][0]*pp[0][1], -pp[1][0]])# pp[0][0] = kpts1[id1].pt[0]
        A.append([0, 0, 0, pp[0][0], pp[0][1], 1, -pp[1][1] * pp[0][0], -pp[1][1]*pp[0][1], -pp[1][1]])
    A = np.array(A)
    #print("A = ",np.array(A))

    # Use svd to calculate vh
    _, _, vh = LA.svd(A, full_matrices=True)

    # Then, select the last singular vector of vh, which respects
    # to the eigenvector of A* x A with smallest eigenvalue
    # for total least squares minimization
    return vh[-1].reshape(3, 3)


def manhattanDistance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def euclidDistance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)



# Tìm ra Homo sao cho với homo đó thì số inliner là Max
# chuyển từ [x,y,w] --> [x/w , y/w]
def getBestHomography(kps1, kps2, matchedDescriptors, pairCoords, threshold=5, iteration=100):
    ''' RANSAC algorithm'''
    bestCntInliers = -float('inf')
    bestPairPts = []
    bestHomography = None
    for i in range(iteration): # cho loop chạy đủ lớn ( == 100 ) thì thuật toán dừng
        homography = estimateHomographyMatrix(kps1, kps2, matchedDescriptors, pairCoords)
        #print(homography)

        cntInliners = 0
        pairPts = []

        #tìm trong pairCoords
        for pcoord in pairCoords: #((132, 238), (222, 251))
            pt = np.array([pcoord[0][0], pcoord[0][1], 1]) # [x, y, 1] của kp1
            pt = pt.reshape(3, 1) # 3 dòng 1 cột
            #-------------------------------------------------
            #_pt = H x pt
            _pt = np.dot(homography, pt) # nhân 2 ma trận 3x3 3x1 = 3x1  *

            # chuyển từ [x,y,w] --> [x/w , y/w]
            _pt = (int(_pt.item(0) / _pt.item(2)),# (132, 238)
                   int(_pt.item(1) / _pt.item(2)))
            # tính khoảng cách từ kp1 vừa Homo --> kp2 của ảnh 2
            dist = euclidDistance(_pt, pcoord[1])

            if dist <= threshold: # đếm số inliner, nếu khoảng cách < ngưỡng thì nó là inliner
                cntInliners += 1
                pairPts.append((pcoord[0], pcoord[1]))

        if cntInliners > bestCntInliers:
            bestCntInliers = cntInliners
            bestHomography = homography
            bestPairPts = pairPts

    print ('#inliers between 2 images: {}'.format(bestCntInliers))

    return bestHomography, bestPairPts # bestPairPts()

# line nối 2 inliner
def drawMatchedPts(image1, image2, pairPts):
    outputImage = concatImages(image1, image2)
    for pt1, pt2 in pairPts:
        p2 = (pt2[0] + image1.shape[1], pt2[1])
        cv2.line(outputImage, pt1, p2, (0, 255, 0))
    return outputImage


if __name__ == '__main__':
    img1 = cv2.imread('./img/test4/1.jpg')
    img2 = cv2.imread('./img/test4/2.jpg')

    img1 = cv2.resize(img1, (0, 0), fx=0.8, fy=0.8)
    img2 = cv2.resize(img2, (0, 0), fx=0.8, fy=0.8)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    kps1, des1 = extractSIFTFeature(gray1)
    kps2, des2 = extractSIFTFeature(gray2)

    matchedDescriptors = getMatchedDescriptors(des1, des2)

    pairCoords = getRelevantPairCoord(kps1, kps2, matchedDescriptors) # lấy tọa độ của 2 keypoints tốt nhất

    h, pairPts = getBestHomography(kps1, kps2, matchedDescriptors, pairCoords) # 

    output = drawMatchedPts(img1, img2, pairPts)

    cv2.imshow('output_ransac', output)
    #cv2.imwrite('./Output/output_ransac.png', output)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
