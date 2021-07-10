# USAGE
# python detect_edges_image.py --image images/guitar.jpg

# import the necessary packages
import argparse
import cv2
import os
import numpy as np
import imutils



# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--edgemap", type=str, required=True,
	help="path to input edgemap")
ap.add_argument("-m", "--mergestyle", type=str, required=True,
	help="Merge style")
ap.add_argument("-b", "--background", type=str, required=True,
	help="Merge style")
args = vars(ap.parse_args())




# load our serialized edge detector from disk
# load the input image and grab its dimensions
edgemap = cv2.imread(args["edgemap"])
background = cv2.imread(args["background"])
mergestyle = args["mergestyle"]
#image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
# background = cv2.cvtColor(background, cv2.COLOR_RGB2RGBA)

#image[np.all(image != [0, 0, 0], axis=2)] = [255, 255, 255]
# image[np.all(image == [254, 254, 254], axis=2)] = [255, 255, 255]
# image[np.all(image == [253, 253, 253], axis=2)] = [255, 255, 255]

bh, bw = background.shape[:2]
#edgemap = imutils.resize(edgemap, width=bw)

#print( h , w, "<==>", bh, bw)
previous_bolded = 0
for i in range(bh):
    for j in range(bw):
		# print i,j
		if (mergestyle == 'WhiteBlackEdges') :
			if (edgemap[i, j][0] > EdgeMap_Threshold) or (edgemap[i, j][1] > EdgeMap_Threshold) or (edgemap[i, j][2] > EdgeMap_Threshold):
				if previous_bolded == 1:
					edgemap[i, j] = [ 255, 255, 255]
					previous_bolded = 0
				else:
					edgemap[i, j] = background[i, j]
					previous_bolded = 0
			else:
				# contrast edges
				if (j != 0) and (j != bw-1):
					if (edgemap[i, j+1][0] > EdgeMap_Threshold) or (edgemap[i, j+1][1] > EdgeMap_Threshold) or (edgemap[i, j+1][2] > EdgeMap_Threshold):
						edgemap[i, j] = [ 255, 255, 255]
				else:
					# Bolded
					print edgemap[i, j]
					edgemap[i, j] = [ 0, 0, 0]
					previous_bolded = 1


		elif (mergestyle == 'BlackEdges'):
			EdgeMap_Threshold = 150
			if (edgemap[i, j][0] > EdgeMap_Threshold) or (edgemap[i, j][1] > EdgeMap_Threshold) or (edgemap[i, j][2] > EdgeMap_Threshold):
				edgemap[i, j] = background[i, j]
			else :
				edgemap[i, j] = [ 0, 0, 0]

		elif (mergestyle == 'WhiteEdges'):
			if (edgemap[i, j][0] < EdgeMap_Threshold) or (edgemap[i, j][1] < EdgeMap_Threshold) or (edgemap[i, j][2] < EdgeMap_Threshold):
				edgemap[i, j] = background[i, j]
			else :
				edgemap[i, j] = [ 255, 255, 255]
				#print(image[i, j])

		elif (mergestyle == 'WhiteShade'):
			if (edgemap[i, j][0] > EdgeMap_Threshold) or (edgemap[i, j][1] > EdgeMap_Threshold) or (edgemap[i, j][2] > EdgeMap_Threshold):
				edgemap[i, j] = background[i, j]
			else :
				shade_offset = 30
				shaded_pixel = [ max(0, min(background[i, j][0] + shade_offset,255)) , max(0, min(background[i, j][1] + shade_offset, 255)) ,max(0, min(background[i, j][2] + shade_offset,255)) ]
				edgemap[i, j] = shaded_pixel
				print background[i, j], "--> " , shaded_pixel

		elif (mergestyle == 'BlackShade'):
			if (edgemap[i, j][0] > EdgeMap_Threshold) or (edgemap[i, j][1] > EdgeMap_Threshold) or (edgemap[i, j][2] > EdgeMap_Threshold):
				edgemap[i, j] = background[i, j]
			else :
				shade_offset = 75
				shaded_pixel = [ max(0,background[i, j][0] - shade_offset) , max(0,background[i, j][1] - shade_offset) ,max(0,background[i, j][2] - shade_offset) ]
				edgemap[i, j] = shaded_pixel


# alpha = 0.8
# beta = 0.4
#beta = 1 - alpha
#(hedH, hedW) = hed.shape[:2]
#(bckdH, bckdW) = background.shape[:2]
#print("[INFO] HED dimensions:", hedH, hedW, "background dimensions:",bckdH, bckdW)

#added_image = cv2.addWeighted(background,alpha,image,beta,1)
#added_image = cv2.add(background, image)
# cv2.imwrite("/FadingMemory/images/edges.jpg", image)
# cv2.imwrite("/FadingMemory/images/background.jpg", background)
# cv2.imwrite("/FadingMemory/images/output_image.jpg", added_image)
mergedimage = edgemap
cv2.imwrite("/FadingMemory/images/output_image.jpg", mergedimage)
