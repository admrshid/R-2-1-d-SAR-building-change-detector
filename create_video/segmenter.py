""" For a given image splits it further into segments """
from affine import Affine
import numpy as np
import rasterio as r

class split_label():
    """ 
    Splits the x and y dimension further into two, resulting in four segmented areas.
    This is to prevent small scale features to be lost during NN training
    """

    def __init__(self, path, segments_x:int, segments_y:int):

        if not path.endswith(".tiff"):
            raise RuntimeError("File is not .tiff")
        
        self.path = path
        self.segments_x = segments_x
        self.segments_y = segments_y

    def get_segments(self):
        
        with r.open(self.path, 'r') as f:
            img = f.read()
            meta = f.meta

            if img.ndim == 2 or img.ndim == 3:
                if img.ndim == 3 and np.squeeze(img).ndim ==3:
                    raise RuntimeError("channel dimension more than one")
                if img.ndim == 3:
                    img_new = np.squeeze(img)
            else:
                raise RuntimeError("ndim of .tiff not 3 or 2")

            x_length = meta["width"]
            y_length = meta["height"]
            transform_old = meta["transform"]

            if x_length % self.segments_x != 0:
                raise RuntimeError("Cannot split x dim into equal segments")
            if y_length % self.segments_y != 0:
                raise RuntimeError("Cannot split y dim into equal segments")
            
            segment_x_length, segment_y_length = x_length / self.segments_x, y_length / self.segments_y

            self.segments = []
            self.transform = []
            ref_x = 0
            for i in range(self.segments_x):
                ref_y = 0
                for j in range(self.segments_y):
                    # split into segments
                    segment = img_new[int(ref_y):int(ref_y+segment_y_length), int(ref_x):int(ref_x+segment_x_length)]
                    # get new affine transformation
                    updated_transform = transform_old*Affine.translation(ref_x,ref_y)
                    
                    self.transform.append(updated_transform)
                    self.segments.append(segment)
                    ref_y += segment_y_length
                ref_x += segment_x_length