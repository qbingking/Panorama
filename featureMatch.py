import cv2
import numpy as np

# Lấy key,des SIFT của 1 hình gray
def extractSIFTFeature(grayscaleImage):
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(grayscaleImage, None)
    return keypoints, descriptors

# khoảng cách bằng căn giữa 2 vecs ( des1- des2)^2
def euclidDistance(descriptor1, descriptor2):
    return np.sqrt(np.sum((descriptor1 - descriptor2) ** 2))

# với mỗi des 1 trong ảnh 1 -- tìm khoảng cách tốt nhất -- Des chứa Mô tả của nhiều keypoint mỗi key point là 1 vector 128
def getDescriptorsDistance(descriptors1, descriptors2):
    desPairDist = []
    #print('xxxx',len(descriptors1))
    for id1, des1 in enumerate(descriptors1): #  list of tuple (count, value) == (id1, des1),  id1 là id của keypoint
        bestId = 0
        bestDistance = float('inf') # infinity vô cực 
        # secondBestId = bestId
        secondBestDistance = bestDistance
        for id2, des2 in enumerate(descriptors2):
            dis = euclidDistance(des1, des2) # khoảng cách giữa 2 vec
            #print("des1",[des1])
            #break
            if dis < bestDistance: # tìm khoảng cách nhỏ nhất
                secondBestDistance = bestDistance
                bestId = id2
                bestDistance = dis
                #if( secondBestDistance == float('inf')):
                    #print("secondBestDistance:",secondBestDistance)

        # (Distance, id1, id2)
        desPairDist.append((bestDistance / secondBestDistance, (id1, bestId))) # bestID = id2 is best distance
    #print("descriptors1",[i for i in descriptors1], len(descriptors1))
    return desPairDist

# lấy những cặp kp mà khoảng cách nhỏ hơn 0.6
def getMatchedDescriptors(descriptors1, descriptors2, threshold=0.6):
    desPairDist = getDescriptorsDistance(descriptors1, descriptors2)
    # print("desPairDist",desPairDist)
    # die()
    sortedDesPairDist = sorted(desPairDist) # Sort theo dis tăng dần
    index = 0
    #đếm số keypoint <= threshold 
    for dis, (id1, id2) in sortedDesPairDist:
        #print("dis",dis) [1,2,44,55,77,78,79] thres = 55 return [1,2,44,55]
        if dis > threshold:
            break
        index += 1
    #print(sortedDesPairDist[:index])
    return sortedDesPairDist[:index]

# nối im1 + im2 = hình lớn
def concatImages(image1, image2): 
    h1, w1, _ = image1.shape # return 3 tham số, lấy 2 cái đầu 
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
    # Read images phải ghi full path
    img1 = cv2.imread('./img/city/2.jpg')
    img2 = cv2.imread('./img/city/1.jpg')

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
    # cv2.imwrite('./Output/output.png', outputImage)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
