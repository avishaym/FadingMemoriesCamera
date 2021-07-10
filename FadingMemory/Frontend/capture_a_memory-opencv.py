import os
import sys
import glob
import subprocess
import yaml
#from hed.utils.io import IO

remove_capture = 0
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
latest_filename = os.path.basename(latest_file)
file_prefix, latest_idx, stageandjpg = latest_filename.split("_")
next_idx = int(latest_idx) + 1
print "latest cature is:", latest_filename, "indexed:", latest_idx
print "next idx:", next_idx
# Capture an images

capture_filename = captures_dir + "/IMG_" + str(next_idx) + "_cap.jpg"
subprocess.call(["/usr/bin/fswebcam","-r","1600x1200",capture_filename])

# Verify image was captured
try:
    os.path.isfile(capture_filename)
    image_captured = 1
except Exception as err:
    print("Failed to capture image by camera")
    sys.exit()


# Generate "Memory" image

subprocess.call(["docker","run","-v","/FadingMemory:/FadingMemory","-v","/etc/localtime:/etc/localtime:ro","-w","/FadingMemory/Backend","hoyledge_opencv_v6","python","generate_hedcv.py"])

# Verify new memory was crated
memories_wildcard = memories_dir + "/*"
list_of_files = glob.glob(memories_wildcard)
latest_file = max(list_of_files, key=os.path.getctime)
latest_filename = os.path.basename(latest_file)
file_prefix, latest_idx, stageandjpg = latest_filename.split("_")
print("latest memory file is:", latest_filename, "indexed:", latest_idx )
if int(latest_idx) != int(next_idx):
    subprocess.call(["rm",capture_filename])
    sys.exit("Failed to generate a new memory")
else:
    print "Memory creation completed successfully!!!"

if remove_capture == 1:
    subprocess.call(["rm",capture_filename])
