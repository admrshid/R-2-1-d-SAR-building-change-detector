""" Combine tiles together into a matrix and also returns longlat dim values with available res from image """

import rasterio as r
import numpy as np

class combine_tile():

    def __init__(self,positional_matrix,path_list):

        self.positional_matrix = positional_matrix
        self.path_list = path_list

    def get_lengths(self):

        self.no_rows, self.no_cols = self.positional_matrix.shape
        self.lengths_x, self.lengths_y = [0],[0]
        
        for j in range(self.no_cols):
            current_tile = self.positional_matrix[0,j]
            current_tile_img = self.path_list[current_tile-1]
            with r.open(current_tile_img, 'r') as f:
                meta = f.meta

                self.lengths_x.append(meta['width'])

        for i in range(self.no_rows):
            current_tile = self.positional_matrix[i,0]
            current_tile_img = self.path_list[current_tile-1]
            with r.open(current_tile_img, 'r') as f:
                meta = f.meta

                self.lengths_y.append(meta['height'])

        self.lengths_x_cumul = [None]*len(self.lengths_x)
        for i in range(len(self.lengths_x)):
            if i == 0:
                self.lengths_x_cumul[i] = self.lengths_x[i]
            else:
                self.lengths_x_cumul[i] = self.lengths_x[i] + self.lengths_x[i-1]

        self.lengths_y_cumul = [None]*len(self.lengths_y)
        for i in range(len(self.lengths_y)):
            if i == 0:
                self.lengths_y_cumul[i] = self.lengths_y[i]
            else:
                self.lengths_y_cumul[i] = self.lengths_y[i] + self.lengths_y[i-1]

    def fill_tile(self):

        # create empty array
        self.full_tile = np.zeros(shape=(sum(self.lengths_y),sum(self.lengths_x)))
        tot = self.no_cols*self.no_rows
        self.transforms = [None]*tot
        
        for i in range(self.no_rows):
            
            for j in range(self.no_cols):
                current_tile = self.positional_matrix[i,j]
                current_tile_img = self.path_list[current_tile-1]
                with r.open(current_tile_img, 'r') as f:
                    img = f.read()
                    meta = f.meta
                    self.transforms[current_tile-1] = meta["transform"]

                    self.full_tile[self.lengths_y_cumul[i]:self.lengths_y_cumul[i+1],self.lengths_x_cumul[j]:self.lengths_x_cumul[j+1]] = img

    def get_longlat(self):

        self.long_x, self.lat_y = [], []

        # get long_x

        for col in range(self.no_cols):
            current_tile = self.positional_matrix[0,col]
            current_transform = self.transforms[current_tile-1]
            if col != self.no_cols-1:
                for j in range(self.lengths_x[col+1]):
                    xs, _ = current_transform*(j,0)
                    self.long_x.append(xs)
            else:
                for j in range(self.lengths_x[col+1]+1):
                    xs, _ = current_transform*(j,0)
                    self.long_x.append(xs)
        
        self.long_x = np.array(self.long_x)
        # get lat_y

        for row in range(self.no_rows):
            current_tile = self.positional_matrix[row,0]
            current_transform = self.transforms[current_tile-1]
            if row != self.no_rows-1:
                for j in range(self.lengths_y[row+1]):
                    _, ys = current_transform*(0,j)
                    self.lat_y.append(ys)
            else:
                for j in range(self.lengths_y[row+1]+1):
                    _, ys = current_transform*(0,j)
                    self.lat_y.append(ys)
        self.lat_y = np.array(self.lat_y)