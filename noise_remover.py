""" Removes noise from output image """
from skimage.morphology import remove_small_objects
import rasterio as r
import os
import sys

def list_file(full_path):
    list = []
    for root, _, files in os.walk(full_path):
        if "To" in root:
            for file in files:
                if file.endswith('.tiff'):
                    list.append(os.path.join(root,file))
    
    return list

# creates full path
def full_path(folder):
    full_path = os.path.join(os.getcwd(),folder)
    
    return full_path

def getfile(output):

    """ Gets list of absolute paths of images in an output file"""

    # Get full abs path of output folder
    fullpath = full_path(output) 
    
    files = list_file(fullpath)

    return files

def remove_noise(file, output, minsize):

    out_dir = f'{output}_filtered'
    out_path = os.path.join(os.getcwd(),out_dir)
    os.makedirs(out_path, exist_ok=True)
    filename =f"{os.path.basename(file).split('.')[0]}_filtered.tiff"
    file_dir = os.path.join(out_path,filename)

    with r.open(file, 'r') as f:
        img = f.read()
        meta = f.meta

        #Preparing the np data to be fed into remove_small_objects
        img[img == 255.0] = 1 
        img = img.astype(bool)

        #Filter out artefacts 
        img_filtered = remove_small_objects(img, min_size=minsize, connectivity=1)

        #Reconvert to np array of black and white pixel values
        img_filtered = img_filtered.astype(float)
        img_filtered[img_filtered == 1.0] = 255.0 

        with r.open(file_dir, 'w', **meta) as dst:
            dst.write(img_filtered)

if __name__ == "__main__":

    output = sys.argv[1] # output file where the unfiltered files are located
    minsize = sys.argv[2] # threshold for label size to be filtered out
    minsize = int(minsize)

    # Get list of files
    files = getfile(output)
    for file in files:
        remove_noise(file,output,minsize)







