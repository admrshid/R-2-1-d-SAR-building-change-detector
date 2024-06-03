from create_video.sort import sortdata
from create_video.sort import import_data_list
from create_video.sort import filter_unopenable_imgs
from create_video.sort import align_videoframes
from create_video.vid_builder import build_vid
from create_video.saver import createpath
from create_video.segmenter import split_label
from create_video.detector import detect_white_pixel
from pathlib import Path
import os
import sys

def main(path,segment_dec):

    if segment_dec == "Yes":
        to_segment = True
    elif segment_dec == "No":
        to_segment = False

    path = Path(path)

    ### STEP 1 : Sort the files 

    metrics = ['VH_COHERENCE','VH','VV_COHERENCE','VV']

    # get list of .tif files
    files = import_data_list(path)
    # filter +-inf/nan files and save to path\filtered
    filtered = filter_unopenable_imgs(files,path)

    #new_path = os.path.join(path,'filtered') # set path to dir containing filtered images

    sorted_dict = sortdata(filtered,metrics,path)()

    # batch the data to save memory (actually, maybe instead of saving it in a dict, just immediately save it on the disk), because I did not have enough RAM

    ### STEP 2 : Go through labels, for each label and metric build up 3d video

    change_labels = [f for f in os.listdir(os.path.join(os.getcwd(),'change')) if f.endswith('.tiff')]
    no_change_labels = [f for f in os.listdir(os.path.join(os.getcwd(),'no_change')) if f.endswith('.tiff')]

    if to_segment:
        # go through each label and segment it in memory, only store once for each iter

        for metric in metrics:
            for label in change_labels:
                split_path = os.path.join(os.getcwd(),'change',label)
                seg = split_label(split_path,segments_x=2,segments_y=2)
                seg.get_segments()
                count = 1
                for k in range(len(seg.segments)):
                    img = seg.segments[k]
                    transform = seg.transform[k]
                    proceed  = detect_white_pixel(img,0)
                    if proceed == 'no_change': # discard black segmented labels 
                        continue
                    save_dir = os.path.join(path,'video_dataset','change',label.split('.')[0],f'segment_{count}',metric)
                    os.makedirs(save_dir, exist_ok=True)
                    name = label.split(".")[0]
                    label_input = f'{name}_segment_{count}'
                    vid = build_vid(label_input,sorted_dict,metric,'change',save_dir,img,transform,segment=True)
                    vid.buildframes()
                    count += 1

        for metric in metrics:
            for label in no_change_labels:
                split_path = os.path.join(os.getcwd(),'no_change',label)
                seg = split_label(split_path,segments_x=2,segments_y=2)
                seg.get_segments()
                count = 1
                for k in range(len(seg.segments)):
                    img = seg.segments[k]
                    transform = seg.transform[k]
                    save_dir = os.path.join(path,'video_dataset','no_change',label.split('.')[0],f'segment_{count}',metric)
                    os.makedirs(save_dir, exist_ok=True)
                    name = label.split('.')[0]
                    label_input = f'{name}_segment_{count}'
                    vid = build_vid(label_input,sorted_dict,metric,'no_change',save_dir,img,transform,segment=True)
                    vid.buildframes()
                    count += 1
        
        ### STEP 3: for every label need to have all dates existing between them, find intersection through all label and metric. 
        # get all video filenames
        align_videoframes(path,to_segment)
    
    else:

        # go through change labels first
        for metric in metrics:
            for label in change_labels:
                save_dir = createpath(path,'video_dataset','change',label.split('.')[0],metric)
                vid = build_vid(label,sorted_dict,metric,'change',save_dir,None,None)
                vid.buildframes()

        # go through unchange labels
        for metric in metrics:
            for label in no_change_labels:
                save_dir = createpath(path,'video_dataset','no_change',label.split('.')[0],metric)
                vid = build_vid(label,sorted_dict,metric,'no_change',save_dir,None,None)
                vid.buildframes()


        ### STEP 3: for every label need to have all dates existing between them, find intersection through all label and metric. 
        # get all video filenames
        align_videoframes(path,to_segment)

if __name__ == "__main__":

    path = sys.argv[1]
    segment_dec = sys.argv[2]

    main(path,segment_dec)
