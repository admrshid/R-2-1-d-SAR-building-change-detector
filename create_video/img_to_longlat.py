""" Gets min long, min lat, max long, max lat for a given image """
import rasterio as r
from rasterio.crs import CRS

class coord:
    " Reads a .tif file and extracts its resolution, min lat&long, max lat&long"
    def __init__(self,img_path,segment,transform,segmented_labels=False):
        self.path = img_path
        self.segmented_labels = segmented_labels
        self.segment = segment
        self.transform = transform

    def compute(self):

        long_0, lat_0 = self.transform*(0,0)
        long_1, lat_1 = self.transform*(self.no_pixel_x,self.no_pixel_y) # this works because (row_pixel,col_pixel) is zero indexed
        self.min_long, self.min_lat = min([long_0,long_1]), min([lat_0,lat_1])
        self.max_long, self.max_lat = max([long_0,long_1]), max([lat_0,lat_1])

    def get_coord(self):

        if not self.segmented_labels:

            if self.path.endswith(".tif") or self.path.endswith(".tiff"):
                
                with r.open(self.path, 'r') as f:
                    # Extract relevant info from file, any var which considered relevant to be in memory has self.
                    meta = f.meta

                    # .meta returns a dictionary of info regarding .tif. The transform key holds the Affine object which maps (row_pixel,col_pixel) --> (long,lat)
                    self.transform = meta["transform"]
                    self.crs = meta["crs"]
                    self.no_pixel_x, self.no_pixel_y = meta["width"], meta["height"]
                    
                    # Only accept WGS:84
                    if self.crs != CRS.from_epsg(4326):
                        raise RuntimeError("File crs is not epsg 4326")
                
                    """
                    The Affine object treats the (row_pixel,col_pixel) cartesian datum as the upper left of the img as datum.
                    Increasing x running to the right, increasing y running down, from the datum.
                    Affine object * (row_pixel,col_pixel) = (long,lat) 
                    Coord system = [WGS:84] --> coord system depends on file being read, but for now only accept WGS:84 (need code for reprojection for more general use)
                    """
                    self.compute()

                    return self.min_long, self.min_lat, self.max_long, self.max_lat, self.path

            else:
                raise RuntimeError("File to be read is not .tif or .tiff file")
            
        else:

            # Write out code to read off segmented labels
            self.no_pixel_x = self.segment.shape[1]
            self.no_pixel_y = self.segment.shape[0]
            
            self.compute()

            return self.min_long, self.min_lat, self.max_long, self.max_lat
        
