import os
import numpy as np
import yaml
from time import strftime

# Merge types:
# 1 - Thin black edges
# 2 - Thicker black edges
# 3 - Overlay shadow
# 4 - Original color edges

# Choose background image by current time
# File name and time format expected: <IMGPREFIX>-%Y%m%d-%H%M.fileext

class EdgeMerge():
    def __init__(self, file_idx, config_file):
        try:
            pfile = open(config_file)
            cfgs = yaml.load(pfile, Loader=yaml.FullLoader)
            memories_dir = cfgs['memories_dir']
            edges_dir = cfgs['edges_dir']
            captures_dir = cfgs['captures_dir']
            bckgrnd_dir = cfgs['bckgrnd_dir']
            bckgrnd_prefix = cfgs['bckgrnd_prefix']

            pfile.close()

        except Exception as err:
            print("something")
            #self.io.print_error('Error reading config file {}, {}'.format(config_file), err)

    def Merge():

        HHMM_now = strftime("%H%M")
        HHMx_now = HHMM_now[:-1]
        bckgrnd_filename = bckgrnd_prefix + HHMx_now +"*"
        bckgrnd_fullpath_prefix = os.path.join(bckgrnd_dir, bckgrnd_filename)
        print "looking for background image file starting with:", bckgrnd_fullpath_prefix

##### MAIN #####
if __name__ == '__main__':
