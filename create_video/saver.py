""" Saves a numpy array into tiff """
import os
import rasterio as r
import numpy as np
from rasterio.transform import from_origin

def createpath(parent_dir:str,root:str,type:str,label:str,metric:str):

    # create path for class
    typedir = os.path.join(parent_dir,root,type,label,metric)
    os.makedirs(typedir,exist_ok=True)

    return typedir

def save(img,long,lat,savepath):

    top_left_lon = long.min()
    top_left_lat = lat.max()
    pixel_width = abs(long[0] - long[1])
    pixel_height = abs(lat[0] - lat[1])

    transform = from_origin(top_left_lon,top_left_lat, pixel_width, pixel_height)

    metadata = {
        'driver': 'GTiff',
        'dtype' : img.dtype,
        'count' : 1,
        'width' : img.shape[1],
        'height' : img.shape[0],
        'crs' : 'EPSG:4326',
        'transform' : transform
    }
    img = np.expand_dims(img,axis=0)
    with r.open(savepath, 'w', **metadata) as dst:
        dst.write(img)
