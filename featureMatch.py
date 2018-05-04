import cv2
import numpy as np


def extractSIFTFeature(grayscaleImage):
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(grayscaleImage, None)
    return keypoints, descriptors


def euclidDistance(descriptor1, descriptor2):
    return np.sqrt(np.sum((descriptor1 - descriptor2) ** 2))


def getDescriptorsDistance(descriptors1, descriptors2):
    desPairDist = []
    for id1, des1 in enumerate(descriptors1):
        bestId = 0
        bestDistance = float('inf')
        # secondBestId = bestId
        secondBestDistance = bestDistance
        for id2, des2 in enumerate(descriptors2):
            dis = euclidDistance(des1, des2)
            if dis < bestDistance:
                # secondBestId = bestId
                secondBestDistance = bestDistance

                bestId = id2
                bestDistance = dis

        desPairDist.append((bestDistance / secondBestDistance, (id1, bestId)))
    return desPairDist


def getMatchedDescriptors(descriptors1, descriptors2, threshold=0.6):
    desPairDist = getDescriptorsDistance(descriptors1, descriptors2)
    sortedDesPairDist = sorted(desPairDist)
    index = 0
    for dis, (id1, id2) in sortedDesPairDist:
        if dis > threshold:
            break
        index += 1
    return sortedDesPairDist[:(index + 1)]


def concatImages(image1, image2):
    h1, w1, _ = image1.shape
    h2, w2, _ = image2.shape
    maxHeight = max(h1, h2)
    totalWidth = w1 + w2
    result = np.zeros((maxHeight, totalWidth, 3), dtype=np.uint8)
    result[:h1, :w1, :] = image1
    result[:h2, w1:(w1 + w2), :] = image2
    return result


def drawMatchedPts(image1, image2, kpts1, kpts2, descriptors1, descriptors2):
    matchedDescriptors = getMatchedDescriptors(descriptors1, descriptors2)
    outputImage = concatImages(image1, image2)
    for _, (id1, id2) in matchedDescriptors:
        pt1 = (int(kpts1[id1].pt[0]), int(kpts1[id1].pt[1]))
        pt2 = (int(kpts2[id2].pt[0] + image1.shape[1]), int(kpts2[id2].pt[1]))
        cv2.line(outputImage, pt1, pt2, (0, 255, 0))
    return outputImage


def normalizeVector(vector):
    return vector / np.sum(vector)


if __name__ == "__main__":
    # Read images
    img1 = cv2.imread('./Pictures/a.png')
    img2 = cv2.imread('./Pictures/b.png')

    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Extract key points and features descriptors
    kp1, des1 = extractSIFTFeature(gray1)
    kp2, des2 = extractSIFTFeature(gray2)

    # Normalize features descriptors
    des1 = np.array([normalizeVector(vector) for vector in des1])
    des2 = np.array([normalizeVector(vector) for vector in des2])

    # Write output image
    outputImage = drawMatchedPts(img1, img2, kp1, kp2, des1, des2)
    cv2.imshow('figure', outputImage)
    cv2.imwrite('./Output/output.png', outputImage)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
