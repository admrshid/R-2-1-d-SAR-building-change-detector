"""Detects presence of white pixels"""
import numpy as np

def detect_white_pixel(img, tolerance):
    css = img.reshape(-1)
    unique, counts = np.unique(css, return_counts=True)
    if len(unique) > 2:
        print("Warning, not binary image")
        return
    
    if len(unique) == 2:
    
        if unique[1] != 255.0 or unique[0] != 0.0:
            print("Warning, not black and white image")
            return
        
        if counts[1] > tolerance:
            out = 'change'
            return out
        else:
            out = 'no_change'
            return out
    else:

        if unique[0] == 0.0:
            out = 'no_change'
            return out
        elif unique[0] == 255.0:
            out = 'change'
            return out

        print('Warning: not black or white image')
        return




