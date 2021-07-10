import os
import sys
import glob
import subprocess
import os
import sys
import argparse
import yaml
import cv2
import imutils
import numpy as np
import logging
from datetime import datetime
import sqlite3

def alignEdges(captured, bckgrnd, edges):

  MAX_FEATURES = 500
  GOOD_MATCH_PERCENT = 0.15


  # Convert images to grayscale
  capGray = cv2.cvtColor(captured, cv2.COLOR_BGR2GRAY)
  bckgrndGray = cv2.cvtColor(bckgrnd, cv2.COLOR_BGR2GRAY)
  edgesGray = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)
  # Detect ORB features and compute descriptors.
  orb = cv2.ORB_create(MAX_FEATURES)
  keypoints1, descriptors1 = orb.detectAndCompute(capGray, None)
  keypoints2, descriptors2 = orb.detectAndCompute(bckgrndGray, None)

  # Match features.
  matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
  matches = matcher.match(descriptors1, descriptors2, None)

  # Sort matches by score
  matches.sort(key=lambda x: x.distance, reverse=False)

  # Remove not so good matches
  numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
  matches = matches[:numGoodMatches]

  # # Draw top matches
  # imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
  # cv2.imwrite("matches.jpg", imMatches)

  # Extract location of good matches
  pointsCap = np.zeros((len(matches), 2), dtype=np.float32)
  pointsBckgrnd = np.zeros((len(matches), 2), dtype=np.float32)

  for i, match in enumerate(matches):
    pointsCap[i, :] = keypoints1[match.queryIdx].pt
    pointsBckgrnd[i, :] = keypoints2[match.trainIdx].pt

  # Find homography
  try:
      h, mask = cv2.findHomography(pointsCap, pointsBckgrnd, cv2.RANSAC)
  except Exception as err:
      sys.exit("Failed to find homography - image taken too different from background")


  # Use homography
  height, width, channels = bckgrnd.shape
  edgesReg = cv2.warpPerspective(edges, h, (width, height))

  return edgesReg, h

def alignEdges_FLANN(captured, bckgrnd, edges):

    MIN_MATCH_COUNT = 3

    # Convert images to grayscale
    capGray = cv2.cvtColor(captured, cv2.COLOR_BGR2GRAY)
    bckgrndGray = cv2.cvtColor(bckgrnd, cv2.COLOR_BGR2GRAY)
    edgesGray = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    capkp, capdes = sift.detectAndCompute(capGray,None)
    bckgrndkp, bckgrnddes = sift.detectAndCompute(bckgrndGray,None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(capdes,bckgrnddes,k=2)
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
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ capkp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ bckgrndkp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h, w = bckgrndGray.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        edgesRes1 = cv2.polylines(edges,[np.int32(dst)],True,255,3, cv2.LINE_AA)
        edgesReg = cv2.warpPerspective(edges, M, (w, h))

    else:
        print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        matchesMask = None

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)
    # Write control matches image
    img3 = cv2.drawMatches(capGray,capkp,bckgrndGray,bckgrndkp,good,None,**draw_params)
    cv2.imwrite('/FadingMemory/images/FLANN_matches_control.jpg',img3)

    return edgesReg, M

def FMDB_getlastid():
    conn = sqlite3.connect('/FadingMemory/Backend/FMDB/FMDB.db')
    c = conn.cursor()
    c.execute("SELECT id FROM images_metadata ORDER BY id DESC LIMIT 1")
    lastid = c.fetchone()[0]
    conn.close()
    return lastid


class CropLayer(object):
        def __init__(self, params, blobs):
                # initialize our starting and ending (x, y)-coordinates of
                # the crop
                self.startX = 0
                self.startY = 0
                self.endX = 0
                self.endY = 0

        def getMemoryShapes(self, inputs):
                # the crop layer will receive two inputs -- we need to crop
                # the first input blob to match the shape of the second one,
                # keeping the batch size and number of channels
                (inputShape, targetShape) = (inputs[0], inputs[1])
                (batchSize, numChannels) = (inputShape[0], inputShape[1])
                (H, W) = (targetShape[2], targetShape[3])

                # compute the starting and ending crop coordinates
                self.startX = int((inputShape[3] - targetShape[3]) / 2)
                self.startY = int((inputShape[2] - targetShape[2]) / 2)
                self.endX = self.startX + W
                self.endY = self.startY + H

                # return the shape of the volume (we'll perform the actual
                # crop during the forward pass
                return [[batchSize, numChannels, H, W]]

        def forward(self, inputs):
                # use the derived (x, y)-coordinates to perform the crop
                return [inputs[0][:, :, self.startY:self.endY,
                                self.startX:self.endX]]


def get_latest_filename(dir):

    latest_idx = FMDB_getlastid()
    full_filename = dir + "/IMG_" + str(latest_idx) + "_cap.jpg"
    dir_wildcard = captures_dir + "/*"
    # latest_file_fullpath = max(list_of_files, key=os.path.getctime)
    # list_of_files = glob.glob(dir_wildcard)

    print "Latest file in directory ",dir," is named ", latest_file_fullpath
    file_basename = os.path.basename(latest_file_fullpath)
    IMG, idx, type_ext = file_basename.split("_")
    return latest_file_fullpath, idx

##### MAIN #####
if __name__ == '__main__':


    base_dir = '/FadingMemory'
    config_file = os.path.join(base_dir,"fadingmemory_config.yaml")
    try:
        pfile = open(config_file)
        cfgs = yaml.load(pfile, Loader=yaml.FullLoader)
        memories_dir = cfgs['memories_dir']
        edges_dir = cfgs['edges_dir']
        captures_dir = cfgs['captures_dir']
        bckgrnd_dir = cfgs['bckgrnd_dir']
        bckgrnd_prefix = cfgs['bckgrnd_prefix']
        mergestyle = cfgs["mergestyle"]
        grayscale_background = cfgs["grayscale_background"]
        log_level = cfgs['log_level']

        pfile.close()

    except Exception as err:
        sys.exit("Error reading config file")

    # Setup logger

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('hedcv')
    logger.setLevel(log_level)
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    fh = logging.FileHandler(r'/FadingMemory/fadingmemory.log')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info("-------------Generate_hedcv.py STARTED----------------")

    # memories_wildcard = memories_dir + "/*"
    # list_of_files = glob.glob(memories_wildcard)
    # latest_file = max(list_of_files, key=os.path.getctime)
    # latest_filename = os.path.basename(latest_file)
    # file_prefix, latest_idx, stageandjpg = latest_filename.split("_")


    latest_idx = FMDB_getlastid()
    next_idx = latest_idx + 1
    logger.debug("latest memory image is indexed: %s", latest_idx)
    expected_capture_filename = "IMG_" + str(next_idx) +"_cap.jpg"
    expected_capture_fullname = os.path.join(captures_dir,expected_capture_filename)
    logger.debug("Looking for captured image: %s", expected_capture_fullname)
    if os.path.isfile(expected_capture_fullname):
        src_image = expected_capture_fullname
        idx = next_idx
    else:
        sys.exit("Expected capture image not found")

    #### HED

    logger.debug("loading edge detector")
    protoPath = os.path.sep.join([cfgs["edge_detector"], "deploy.prototxt"])
    modelPath = os.path.sep.join([cfgs["edge_detector"],"hed_pretrained_bsds.caffemodel"])
    net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    cv2.dnn_registerLayer("Crop", CropLayer)

    # Get last capture index number
    logger.debug("Locating latest image captured")
    image = cv2.imread(src_image)
    #(origH, origW) = image.shape[:2]
    #print("[INFO] resizing to 1500...")
    #image = imutils.resize(image, 1500)
    (H, W) = image.shape[:2]
    # construct a blob out of the input image for the Holistically-Nested
    # Edge Detector
    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(W, H),
        mean=(104.00698793, 116.66876762, 122.67891434), swapRB=False, crop=False)
    # set the blob as the input to the network and perform a forward pass
    # to compute the edges
    logger.debug("performing holistically-nested edge detection...")
    logger.debug("setting net input...")
    net.setInput(blob)
    logger.debug("forwarding...")
    edgemap = net.forward()
    logger.debug("resizing ...")
    edgemap = cv2.resize(edgemap[0, 0], (W, H))
    edgemap = (255 * edgemap).astype("uint8")
    edgemap = cv2.cvtColor(edgemap, cv2.COLOR_GRAY2RGB)
    # edgemap = cv2.bitwise_not(edgemap)
    # Write edgemap edgemap to file
    edgefile_basename = "IMG_" + str(idx) + "_edges.jpg"
    edgefile_fullpath = os.path.join(edges_dir, edgefile_basename)
    # logger.info("Writing edges image to %s", edgefile_fullpath)
    # cv2.imwrite(edgefile_fullpath, edgemap)


    # Locate background
    ##############################
    if cfgs["bckgrnd_file"] != "AUTO" :
        latest_bckgrnd_fullpath = cfgs["bckgrnd_file"]
    else:
        now = datetime.now()
        HHMM_now = now.strftime("%H%M")
        HHMx_now = HHMM_now[:-1]
        bckgrnd_filename = bckgrnd_prefix + HHMx_now +"*"
        bckgrnd_fullpath_prefix = os.path.join(bckgrnd_dir, bckgrnd_filename)
        logger.info("looking for background image file starting with: %s", bckgrnd_fullpath_prefix)
        list_of_files = glob.glob(bckgrnd_fullpath_prefix)

        try:
            latest_bckgrnd_fullpath = max(list_of_files, key=os.path.getctime)
        except:
            logger.error("failed to find minutly background, trying hourly search")
            HHxx_now = HHMx_now[:-1]
            bckgrnd_filename = bckgrnd_prefix + HHxx_now +"*"
            bckgrnd_fullpath_prefix = os.path.join(bckgrnd_dir, bckgrnd_filename)
            logger.info("looking for background image file starting with: %s", bckgrnd_fullpath_prefix)
            list_of_files = glob.glob(bckgrnd_fullpath_prefix)
            latest_bckgrnd_fullpath = max(list_of_files, key=os.path.getctime)

    logger.info("Background selected is %s", latest_bckgrnd_fullpath)

    background = cv2.imread(latest_bckgrnd_fullpath)

    logger.info("Writing edges image to edgebeforealigncontrol")
    cv2.imwrite("/FadingMemory/images/edgebeforealigncontrol.jpg", edgemap)
    # Align edges according to background

    logger.info("Aligning Edges according to background")
    # edgemap, h = alignEdges(image, background, edgemap)
    edgemap, h = alignEdges_FLANN(image, background, edgemap)
    edgemap = cv2.bitwise_not(edgemap)

    # edgemap = cv2.cvtColor(edgemap, cv2.COLOR_GRAY2RGB)

    logger.info("Writing edges image to %s", edgefile_fullpath)
    cv2.imwrite(edgefile_fullpath, edgemap)

    ################################
    # Start of merger & edge styling
    ################################

    previous_bolded = 0
    EdgeMap_Threshold = 150

    if (grayscale_background == 1):
        logger.info("Converting background to grayscale and saving control image")
        background = cv2.cvtColor(background, cv2.COLOR_RGB2GRAY)
        cv2.imwrite(r"/FadingMemory/images/BW_background_control.jpg", background)
        logger.debug("Forcing merge style to BlackEdges")
        mergestyle = "BlackEdges"



    # Merging according to Style
    ############################
    logger.info("Merging edges to background")

    for i in range(H):
        for j in range(W):
    		# print edgemap[i,j]
    		if (mergestyle == 'WhiteBlackEdges'):
                #logger.info("Style applied WBedges")
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
    					#print edgemap[i, j]
    					edgemap[i, j] = [ 0, 0, 0]
    					previous_bolded = 1


    		elif (mergestyle == 'BlackEdges'):
    			if (edgemap[i, j][0] > EdgeMap_Threshold) or (edgemap[i, j][1] > EdgeMap_Threshold) or (edgemap[i, j][2] > EdgeMap_Threshold):
    				edgemap[i, j] = background[i, j]
    			else :
    				edgemap[i, j] = [ 0, 0, 0]

    		elif (mergestyle == 'WhiteEdges'):
                #logger.info("Style applied: %s", mergestyle)
    			if (edgemap[i, j][0] < EdgeMap_Threshold) or (edgemap[i, j][1] < EdgeMap_Threshold) or (edgemap[i, j][2] < EdgeMap_Threshold):
    				edgemap[i, j] = background[i, j]
    			else :
    				edgemap[i, j] = [ 255, 255, 255]
    				#print(image[i, j])

    		elif (mergestyle == 'WhiteShade'):
                #logger.info("Style applied: %s", mergestyle)
    			if (edgemap[i, j][0] > EdgeMap_Threshold) or (edgemap[i, j][1] > EdgeMap_Threshold) or (edgemap[i, j][2] > EdgeMap_Threshold):
    				edgemap[i, j] = background[i, j]
    			else :
    				shade_offset = 30
    				shaded_pixel = [ max(0, min(background[i, j][0] + shade_offset,255)) , max(0, min(background[i, j][1] + shade_offset, 255)) ,max(0, min(background[i, j][2] + shade_offset,255)) ]
    				edgemap[i, j] = shaded_pixel
    				#print background[i, j], "--> " , shaded_pixel

    		elif (mergestyle == 'BlackShade'):
                #logger.info("Style applied: %s", mergestyle)
    			if (edgemap[i, j][0] > EdgeMap_Threshold) or (edgemap[i, j][1] > EdgeMap_Threshold) or (edgemap[i, j][2] > EdgeMap_Threshold):
    				edgemap[i, j] = background[i, j]
    			else :
    				shade_offset = 75
    				shaded_pixel = [ max(0,background[i, j][0] - shade_offset) , max(0,background[i, j][1] - shade_offset) ,max(0,background[i, j][2] - shade_offset) ]
    				edgemap[i, j] = shaded_pixel

    font = cv2.FONT_HERSHEY_SIMPLEX
    logger.info("Adding text to image")
    cv2.putText(edgemap,'Remember Burn in Motion 2019?', (10,50), font, 1,(0,0,0),3,2)

    memoryfile_basename = "IMG_" + str(idx) + "_memory.jpg"
    memory_fullpath = os.path.join(memories_dir, memoryfile_basename)
    logger.info("Writing memory to: %s", memory_fullpath)
    cv2.imwrite(memory_fullpath, edgemap)

    # logger.info("Writing image to FMDB")
    # conn = sqlite3.connect('/FadingMemory/Backend/FMDB/FMDB.db')
    # c = conn.cursor()
    # c.execute("INSERT INTO images (id, image) VALUES(?, ?)", (idx, buffer(edgemap)))
    # conn.commit()
    # conn.close()

    logger.info("generate_hedcv completed")
