""" Consists of class which gets configs of a group of training tiles """
import numpy as np

class overlap_config():
    """ Checks for a given collection of training SAR data, how can it fit together as a piece (i.e. merge) """
    def __init__(self,list_of_SAR_coords:list):

        self.list_of_SAR_coords = list_of_SAR_coords

    def get_global_minmax(self):
        
        # Get min, max for lat and long for the whole collection of overlapping training data
        # (min_long,min_lat,max_long,max_lat)

        self.global_minmax = []
        for i in range(4):
            temp = []
            for j in self.list_of_SAR_coords:
                temp.append(j[i])
            if i <= 1:
                self.global_minmax.append(min(temp))
            else:
                self.global_minmax.append(max(temp))
        self.global_minmax = tuple(self.global_minmax)

    def get_signature(self):

        # get the relative positions of a reference img to others
        no_imgs = len(self.list_of_SAR_coords)
        self.emp_full = []
        for ref in range(no_imgs):
            ref_array = np.array(self.list_of_SAR_coords[ref])
            ref_array = np.stack((ref_array,ref_array,ref_array,ref_array),axis=1)
            
            emp = np.zeros(shape=(3,3))
            for secondary in range(no_imgs):
                if secondary != ref:
                    secondary_array = np.array(self.list_of_SAR_coords[secondary])
                    secondary_array = np.stack((secondary_array,secondary_array,secondary_array,secondary_array),axis=0)
                    self.signature = ref_array - secondary_array
                    print(ref_array)
                    print(secondary_array)
                    print(self.signature)

                    # create an empty matrix 3by3 to place the relative positions
                    emp[1,1] = ref+1
                    # place in the secondary images wrt to ref according to signature
                    tol = 0.001
                    if abs(self.signature[0,0]) < tol and abs(self.signature[2,2]) < tol and abs(self.signature[1,3]) < tol:
                        emp[2,1] = secondary+1
                    elif abs(self.signature[0,0]) < tol and abs(self.signature[2,2]) < tol and abs(self.signature[3,1]) < tol:
                        emp[0,1] = secondary+1
                    elif abs(self.signature[1,1]) < tol and abs(self.signature[3,3]) < tol and abs(self.signature[2,0]) < tol:
                        emp[1,2] = secondary+1
                    elif abs(self.signature[1,1]) < tol and abs(self.signature[3,3]) < tol and abs(self.signature[0,2]) < tol:
                        emp[1,0] = secondary+1
            self.emp_full.append(emp)