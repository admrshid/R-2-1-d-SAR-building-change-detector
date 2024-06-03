""" Builds up a video for a certain label and metric """
from create_video.img_to_longlat import coord
from create_video.overlap_finder import overlap_training_SAR_data
from create_video.checks import within_boundaries
from create_video.checks import check_global_box
from create_video.checks import merge_tiles_position
from create_video.overlap_config import overlap_config
from create_video.combine_tile import combine_tile
from create_video.sort import get_indexes
from create_video.saver import save
from create_video.sort import get_date
from create_video.sort import filter_infnan_imgs_raw
import numpy as np
import os

def submatrix(label_latlong,longs,lats,grouped_tile):

    label_min_long, label_min_lat, label_max_long, label_max_lat = label_latlong[0], label_latlong[1], label_latlong[2], label_latlong[3]
    ind_x = (longs >= label_min_long) & (longs <= label_max_long)
    ind_y = (lats >= label_min_lat) & (lats <= label_max_lat)
                
    inds_x = get_indexes(ind_x,True) # this returns a list of indexes
    inds_y = get_indexes(ind_y,True)

    longitude = longs[inds_x[0]:inds_x[-1]]
    latitude = lats[inds_y[0]:inds_y[-1]]
    frame_array = grouped_tile[inds_y[0]:inds_y[-1], inds_x[0]:inds_x[-1]]

    return frame_array, longitude, latitude

class build_vid():

    def __init__(self,label:str,sorted_dict,metric,label_type,save_dir,segment_arr,segment_trans,segment=False):
        
        label_dir = os.path.join(os.getcwd(),label_type)
        self.label = os.path.join(label_dir,label) # this should be path to label img
        self.metric_frame = sorted_dict[metric]
        self.save_dir = save_dir
        self.metric = metric
        self.segment = segment
        self.segment_arr = segment_arr
        self.segment_trans = segment_trans

    def buildframes(self):
    # for each timeframe get the numpy array which intersects
        
        
        for frame in self.metric_frame:

            if isinstance(frame, list):
            # so far not have dealt with tiles, but just in case

                date = get_date(frame[0]) # dates for all tiles should be the same
                date = date.strftime('%Y_%m_%d')
                savepath = os.path.join(self.save_dir,f'{date}.tiff')

                if os.path.exists(savepath):
                    print(f'{savepath} already exists, skipping')
                    return
                
                print(f"Running frame: {date} for label: {self.label}, metric: {self.metric}")

                tiles_latlong = []
                for tile in frame:
                    
                    temp = coord(tile,None,None)
                    tempo = temp.get_coord()
                    tiles_latlong.append(tempo)

                if not self.segment:
                    temp = coord(self.label,None,None)
                    tempo = temp.get_coord()
                    label_latlong = tempo
                else:
                    temp = coord(None,self.segment_arr,self.segment_trans,segmented_labels=True)
                    tempo = temp.get_coord() # doesn't output path
                    label_latlong = tempo

                # now only get ones which overlap with the label
                rel_tiles = overlap_training_SAR_data(label_latlong,tiles_latlong)
                rel_tiles.get_training_data() # execute function to get overlapping tiles
                rel_tile_latlong = rel_tiles.training_data # list of rel training coords
                rel_tile_path = rel_tiles.training_data_path # list of rel training paths
                if rel_tile_latlong == []:
                    print(f"No data available for label: {self.label}, metric: {self.metric} in frame: {date}")
                    continue

                # following dynamic programming principles, always make checks that supersedes

                ### 1st check : is label within global box of tiles

                overlapconfigs = overlap_config(rel_tile_latlong)
                overlapconfigs.get_global_minmax() # get the global bounds
                global_bounds = overlapconfigs.global_minmax

                check_1 = within_boundaries(label_latlong,global_bounds)
                if not check_1:
                    print(f"No data available for label: {self.label}, metric: {self.metric} in frame: {date}")
                    continue
                
                ### 2nd check : see whether in the list of tiles there is such a tile which shares global boundary, while simultaneously get corner tiles if pass check

                check_2 = check_global_box(rel_tile_latlong,global_bounds,self.label,date)
                if check_2 == False:
                    print(f"No data available for label: {self.label}, metric: {self.metric} in frame: {date}")
                    continue
                else:
                    corner_tile = check_2
                
                ### 3rd check : check for holes within tiles, while simultaneously get the position tile matrix if pass check

                check_3 = merge_tiles_position(corner_tile,rel_tile_latlong)
                if check_3 == False:
                    print(f"No data available for label: {self.label}, metric: {self.metric} in frame: {date}")
                    continue
                else:
                    position_matrix = check_3

                # after passing all the checks, merge the tiles together

                whole_tile = combine_tile(position_matrix,rel_tile_path)
                whole_tile.get_lengths() # get lengths x and y to initiate empty matrix
                whole_tile.fill_tile() # fill in the empty matrix with values from ind tiles according to positional matrix
                whole_tile.get_longlat() # get the longlat values

                grouped_tile = whole_tile.full_tile
                longs = whole_tile.lengths_x
                lats = whole_tile.lengths_y

                # get the submatrix which intersects with label coords

                submat = submatrix(label_latlong,longs,lats,grouped_tile)

                # filter out +-inf/nan values
                filt = filter_infnan_imgs_raw(submat[0])
                if filt == None:
                    print(f'Has +-inf/nan values for label: {self.label}, metric: {self.metric} in frame: {date}')
                    continue

                print(f'Saving {date}.tiff to {savepath}')
                save(submat[0],submat[1],submat[2],savepath)

            # if only one tile do simpler compute
            else:

                date = get_date(frame)
                date = date.strftime('%Y_%m_%d')
                savepath = os.path.join(self.save_dir,f'{date}.tiff')

                if os.path.exists(savepath):
                    print(f'{savepath} already exists, skipping')
                    return
                
                print(f"Running frame: {date} for label: {self.label}, metric: {self.metric}")

                tiles_latlong = []
                temp = coord(frame,None,None)
                tempo = temp.get_coord()

                tiles_latlong.append(tempo)
                
                if not self.segment:
                    temp = coord(self.label,None,None)
                    tempo = temp.get_coord()
                    label_latlong = tempo
                else:
                    temp = coord(None,self.segment_arr,self.segment_trans,segmented_labels=True)
                    tempo = temp.get_coord() # doesn't output path
                    label_latlong = tempo

                # now only get ones which overlap with the label
                rel_tiles = overlap_training_SAR_data(label_latlong,tiles_latlong)
                rel_tiles.get_training_data() # execute function to get overlapping tiles
                rel_tile_latlong = rel_tiles.training_data # list of rel training coords
                rel_tile_path = rel_tiles.training_data_path # list of rel training paths
                if rel_tile_latlong == []:
                    print(f"No data available for label: {self.label}, metric: {self.metric} in frame: {date}")
                    continue

                # following dynamic programming principles, always make checks that supersedes

                ### 1st check : is label within global box of tiles

                overlapconfigs = overlap_config(rel_tile_latlong)
                overlapconfigs.get_global_minmax() # get the global bounds
                global_bounds = overlapconfigs.global_minmax

                check_1 = within_boundaries(label_latlong,global_bounds)
                if not check_1:
                    print(f"No data available for label: {self.label}, metric: {self.metric} in frame: {date}")
                    continue

                # after passing all the checks, merge the tiles together

                position_matrix = np.array([[1]])
                whole_tile = combine_tile(position_matrix,rel_tile_path)
                whole_tile.get_lengths() # get lengths x and y to initiate empty matrix
                whole_tile.fill_tile() # fill in the empty matrix with values from ind tiles according to positional matrix
                whole_tile.get_longlat() # get the longlat values

                grouped_tile = whole_tile.full_tile
                longs = whole_tile.long_x
                lats = whole_tile.lat_y

                # get the submatrix which intersects with label coords

                submat = submatrix(label_latlong,longs,lats,grouped_tile)

                #filter out +-inf/nan images

                filt = filter_infnan_imgs_raw(submat[0])
                if filt:
                    print(f'Saving {date}.tiff to {savepath}')
                    save(submat[0],submat[1],submat[2],savepath)
                else:
                    print(f'Has +-inf/nan values for label: {self.label}, metric: {self.metric} in frame: {date}')
                    continue



