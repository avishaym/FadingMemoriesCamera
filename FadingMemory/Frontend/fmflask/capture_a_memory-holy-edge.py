import os
#import os.path
import sys
import glob
import subprocess
import yaml
#from hed.utils.io import IO

image_captured = 0
base_dir = "/FadingMemory"
config_file = os.path.join(base_dir,"fadingmemory_config.yaml")
try:
    print("Looking for config file:", config_file)
    pfile = open(config_file)
    cfgs = yaml.load(pfile)
    memories_dir = cfgs['memories_dir']
    print("memories directory is:", memories_dir)
    edges_dir = cfgs['edges_dir']
    captures_dir = cfgs['captures_dir']
    bckgrnd_dir = cfgs['bckgrnd_dir']
    bckgrnd_prefix = cfgs['bckgrnd_prefix']
    pfile.close()

except Exception as err:
    print("something")
            #self.io.print_error('Error reading config file {}, {}'.format(config_file), err)

# Set next capture index number
memories_wildcard = memories_dir + "/*"
list_of_files = glob.glob(memories_wildcard)
latest_file = max(list_of_files, key=os.path.getctime)
file_prefix, latest_idx, stageandjpg = latest_file.split("_")
next_idx = int(latest_idx) + 1

# Capture an images

capture_filename = captures_dir + "/IMG_" + str(next_idx) + "_cap.jpg"
subprocess.call(["/usr/bin/fswebcam",capture_filename])

# Verify image was captured
try:
    os.path.isfile(capture_filename)
    image_captured = 1
except Exception as err:
    print("Failed to capture image by camera")


# Extract edges

subprocess.call(["sudo","docker","run","-v","/FadingMemory:/FadingMemory","-v","/etc/localtime:/etc/localtime:ro","fadingmem"])

# # Verify new edge map was crated
# edges_wildcard = edges_dir + "/*"
# list_of_files = glob.glob(edges_wildcard)
# latest_file = max(list_of_files, key=os.path.getctime)
# file_prefix, latest_idx, stageandjpg = latest_file.split("_")
# if latest_idx != next_idx:
#     edgemap_filename = latest_file
# else:
#     sys.exit("Failed to extract edgemaps by HED container")
