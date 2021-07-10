import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import sys

img2 = cv.imread('/FadingMemory/images/backgrounds/homeniche43.jpg',cv.IMREAD_GRAYSCALE)          # queryImage
img1 = cv.imread('/FadingMemory/images/captures/IMG_169_cap.jpg',cv.IMREAD_GRAYSCALE) # trainImage
# Initiate SIFT detector
sift = cv.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)
# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.3*n.distance:
        matchesMask[i]=[1,0]

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.3*n.distance:
        good.append(m)

MIN_MATCH_COUNT = 10
if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,M)

    img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)

else:
    print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)


#plt.imshow(img3,),plt.show()
cv.imwrite('/FadingMemory/images/FLANN_matches.jpg',img3)

#sys.exit()

## Extract location of good matches
#points1 = np.zeros((len(matches), 2), dtype=np.float32)
#points2 = np.zeros((len(matches), 2), dtype=np.float32)

#for i, match in enumerate(matches):
#  points1[i, :] = kp1[match.queryIdx].pt
#  points2[i, :] = kp2[match.trainIdx].pt

# Find homography
h, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC)

# Use homography
height, width = img1.shape
im1Reg = cv.warpPerspective(img1, h, (width, height))

# Write aligned image to disk.
outFilename = "/FadingMemory/images/aligned_test3.jpg"
print("Saving aligned image : ", outFilename);
cv.imwrite(outFilename, im1Reg)
