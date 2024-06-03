""" List of class and functions which provides the list of checks for saying there is data for label """
import numpy as np
from create_video.overlap_config import overlap_config

def within_boundaries(label_coords,global_boundary):

    """Takes in coords from training SAR data and label data and returns True if label within global boundary else False"""
    # Extract data
    global_min_long, global_min_lat, global_max_long, global_max_lat = global_boundary[0], global_boundary[1], global_boundary[2], global_boundary[3]
    label_min_long, label_min_lat, label_max_long, label_max_lat = label_coords[0], label_coords[1], label_coords[2], label_coords[3]

    min_long_check = global_min_long <= label_min_long <= global_max_long
    max_long_check = global_min_long <= label_max_long <= global_max_long
    min_lat_check = global_min_lat <= label_min_lat <= global_max_lat
    max_lat_check = global_min_lat <= label_max_lat <= global_max_lat

    if min_long_check and max_long_check and min_lat_check and max_lat_check:
        return True
    else:
        return False
    

def check_global_box(list_of_SAR_coord,global_boundary,label,date):

    """Checks if the training tiles can be fitted as a square or rectangle, also returns the corner tiles"""

    corner_tiles = {}
    
    global_min_long, global_min_lat, global_max_long, global_max_lat = global_boundary[0], global_boundary[1], global_boundary[2], global_boundary[3]

    # Look through min_global_long,max_global_lat first and see if there exists a tile that shares these attributes
    corner_tiles["upper_left"] = [a+1 for a in range(len(list_of_SAR_coord)) if list_of_SAR_coord[a][0] == global_min_long and list_of_SAR_coord[a][3] == global_max_lat]
    if len(corner_tiles["upper_left"]) != 1:
        print(f"Either more than one tile shares upper left or empty tile in upper left for {label} in frame {date}")
        return False
    
    # Look through max_global_long,max_global_lat and see if there exists a tile that shares these attributes
    corner_tiles["upper_right"] = [a+1 for a in range(len(list_of_SAR_coord)) if list_of_SAR_coord[a][2] == global_max_long and list_of_SAR_coord[a][3] == global_max_lat]
    if len(corner_tiles["upper_right"]) !=1:
        print(f"Either more than one tile shares upper right or empty tile in upper right for {label} in frame {date}")
        return False

    corner_tiles["lower_left"] = [a+1 for a in range(len(list_of_SAR_coord)) if list_of_SAR_coord[a][0] == global_min_long and list_of_SAR_coord[a][1] == global_min_lat]
    if len(corner_tiles["lower_left"]) !=1:
        print(f"Either more than one tile shares lower left or empty tile in lower left for {label} in frame {date}")
        return False

    corner_tiles["lower_right"] = [a+1 for a in range(len(list_of_SAR_coord)) if list_of_SAR_coord[a][2] == global_max_long and list_of_SAR_coord[a][1] == global_min_lat]
    if len(corner_tiles["lower_right"]) !=1:
        print(f"Either more than one tile shares lower right or empty tile in lower right for {label} in frame {date}")
        return False

    return corner_tiles


def get_relevant_signature(signature):
    bottom = signature[2,1] 
    right = signature[1,2] 
    left = signature[1,0]

    return bottom, right, left

class merge_tiles_position():

    def __init__(self,corner_tiles,list_coords_SAR):

        self.list_coords_SAR = list_coords_SAR
        self.corner_tiles = corner_tiles

    def get_position(self):

        temp = np.zeros(shape = (1,1))
        temp[0,0] = self.corner_tiles["upper_left"][0]

        signature = overlap_config(self.list_coords_SAR)
        signature.get_signature()
        signatures = signature.emp_full
        counter = self.corner_tiles["upper_left"][0]
        print(counter)
        run = 0
        length_row = 0
        
        for sign in range(len(self.list_coords_SAR)):
            # keep running through rows and cols, filling in the relative tiles
            
            bottom, right, left = get_relevant_signature(signatures[int(counter-1)])

            if run % 2 == 0:
                torun = right
                if length_row !=0 and abs(fill) == length_row and torun != 0: # introduce check which says that right must be zero if reached end of run
                    return False
                
            elif run % 2 != 0:
                torun = left
                if length_row !=0 and abs(fill+1) == length_row and torun != 0: # introduce check which says that left must be zero if reached end of run
                    return False
            
            if torun != 0:
                if run == 0:
                    temp = np.concatenate((temp,torun.reshape(1,1)),axis=1)
                    counter = torun
                else:
                    temp[run,fill] = torun
                    counter = torun
                    if run % 2 != 0:
                        fill += -1
                    elif run % 2 == 0:
                        fill += 1

            elif bottom != 0:
                run += 1
                # add a row below temp to start filling in the second row
                length_row = temp.shape[1]
                temp1 = np.zeros(shape=(1,length_row))
                temp = np.concatenate((temp,temp1),axis=0)
                # initiate the new row
                if run % 2 != 0:
                    temp[-1,-1] = bottom
                    fill = -2
                if run % 2 == 0:
                    temp[-1,0] = bottom
                    fill = 1
                counter = bottom
            else:
                if temp[0,0] != self.corner_tiles["upper_left"] or temp[0,-1] != self.corner_tiles["upper_right"] or temp[-1,0] != self.corner_tiles["lower_left"] or temp[-1,-1] != self.corner_tiles["lower_right"]:
                    return False
                else:
                    return temp