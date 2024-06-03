""" Rearranges images into folders according to class """
from osm_changes_class_save.detector import detect_white_pixel
from osm_changes_class_save.create_dir import list_file
from osm_changes_class_save.create_dir import list_file_unfiltered
from osm_changes_class_save.create_dir import full_path
import os
import rasterio as r
import sys


def classsaver(path,threshold,type):

    
    out_dir = ['change','no_change']

    # create new folder for each class
    change_path = os.path.join(os.getcwd(),out_dir[0])
    no_change_path = os.path.join(os.getcwd(),out_dir[1])
    os.makedirs(change_path, exist_ok=True)
    os.makedirs(no_change_path, exist_ok=True)


    if type == 'change':

        path = f"{path}_filtered"
        fullpath = full_path(path)
        files = list_file(path)
        
        for f in files:

            filename = os.path.basename(f)

            # check if file exists
            out = os.path.join(change_path,filename)
            if os.path.exists(out):
                print(f'File: {out} already exists, skipping')
                continue

            # reading file and extracting its contents
            with r.open(f, 'r') as file:
                imarray = file.read()
                meta = file.meta

                #Detect whether class is change or no change
                classes = detect_white_pixel(imarray, tolerance=threshold) 

                #Save if change
                if classes == None:
                    sys.exit(1)
                elif classes == 'change':
                    with r.open(out, 'w', **meta) as dst:
                        dst.write(imarray)
    
    elif type == 'no_change':

        fullpath = full_path(path)
        files = list_file_unfiltered(fullpath)
        for f in files:
            
            # Extract filename for use in saving
            filename = os.path.basename(f)

            # Check if file exists
            out = os.path.join(no_change_path,filename)
            if os.path.exists(out):
                print(f'File: {out} already exists, skipping')
                continue

            # reading file and extracting its contents
            with r.open(f, 'r') as file:
                imarray = file.read()
                meta = file.meta

                #Detect whether class is change or no change
                classes = detect_white_pixel(imarray, tolerance=0)

                #Save if no change
                if classes == None:
                    sys.exit(1)
                elif classes == 'no_change':
                    with r.open(out, 'w', **meta) as dst:
                        dst.write(imarray)

    else:

        sys.exit(1)
        

        


        

 
