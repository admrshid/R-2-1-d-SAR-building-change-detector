""" Sorts the files in a directory into their metrics """
from datetime import datetime
from collections import Counter
import fnmatch
import rasterio as r
import os
import numpy as np

def align_videoframes(path,segment):
    all_frames = []
    req_len = 0
    for dir,root,file in os.walk(os.path.join(path,'video_dataset')):
        if file:
            if not fnmatch.fnmatch(dir.split(os.sep)[-2],'segment*') and not segment:
                for fil in file:
                    all_frames.append(os.path.join(dir,fil))
                req_len += 1
            elif fnmatch.fnmatch(dir.split(os.sep)[-2],'segment*') and segment:
                for fil in file:
                    all_frames.append(os.path.join(dir,fil))
                req_len += 1
    
    base = [os.path.basename(a) for a in all_frames]
    unique = Counter(base)
    unique_dict = dict(unique)
    
    for key,item in unique_dict.items():
        if item != req_len:
            remo = get_indexes(base,key)
            for ind in remo:
                print(f'deleting {all_frames[ind]}, since it does not align')
                os.remove(all_frames[ind])

def import_data_list(path):

    list_all = [f for f in os.listdir(path) if f.endswith(".tif")]
    return list_all

def filter_infnan_imgs(paths:list,dir:str):

    os.makedirs(os.path.join(dir,'filtered'), exist_ok=True)

    for path in paths:
        # skip if file exists
        out = os.path.join(dir,'filtered',path)
        if os.path.exists(out):
            print(f'File {out} exists, skipping')
            continue
        try:
            with r.open(os.path.join(dir,path), 'r') as f:
                img = f.read()
                meta = f.meta
                if np.any(np.isinf(img)) or np.any(np.isneginf(img)) or np.any(np.isnan(img)):
                    print(f'filtering {path}')
                    pass
                else:
                    print(f'copying {path} to {out}')
                    with r.open(out, 'w', **meta) as dst:
                        dst.write(img)
        except r.errors.RasterioIOError as e:
            print(f"Error reading {path}: {e}")
            # Skip to the next file
            continue

def filter_unopenable_imgs(paths:list,dir:str):

    keep = []

    for path in paths:
        # skip if file exists
        try:
            with r.open(os.path.join(dir,path), 'r') as f:
                keep.append(path)
        except r.errors.RasterioIOError as e:
            print(f"Error reading {path}: {e}, filtering")
            # Skip to the next file
            continue

    return keep

def filter_infnan_imgs_raw(img):

    if np.any(np.isinf(img)) or np.any(np.isneginf(img)) or np.any(np.isnan(img)):
        return False
    else:
        return True

def sort_metric(file_list:list, metrics:list, dir:str) -> dict:
    # Create a dictionary to store files corresponding to each metric
    metric_dict = {m: [] for m in metrics}

    # Iterate through files and rearrange them according to metric
    for file_name in file_list:
        for metric in metrics:
            if file_name.startswith(metric):
                metric_dict[metric].append(os.path.join(dir,file_name))
                break  # Once matched, move to the next file

    return metric_dict

def get_date(t:str) -> datetime:

    time = t.split('_')[-1][:-11]
    return datetime.strptime(time, '%Y%m%d')

def get_indexes(lst, element):
    return [index for index, value in enumerate(lst) if value == element]

def sort_files_by_date(date_list:list) -> list:

    sorted_dates = sorted(date_list, key=get_date)
    dates = [get_date(f) for f in sorted_dates]
    counter = Counter(dates)
    unique = dict(counter)
    
    for element, items in unique.items():
        if items != 1:
            indexes = get_indexes(dates, element)
            collect = []
            for index in indexes:
                collect.append(sorted_dates[index])
            sorted_dates[indexes[0]] = collect
            sorted_dates = sorted_dates[:indexes[1]] + sorted_dates[indexes[-1]+1:]
    return sorted_dates

def intersect_date(sorted_dict:dict) -> dict:

    metrics = list(sorted_dict.keys())
    intersected_dict = {m:[] for m in metrics}

    for metric in range(len(sorted_dict)):
        reference_list = sorted_dict[metrics[metric]]
        reference_list = [get_date(i) for i in reference_list]
        reference_metric = metrics[:metric] + metrics[metric+1:] # gets all the other metrics other than metric
        full = {} # one to store date
        for f in reference_metric:
            
            full[f] = [get_date(i) for i in sorted_dict[f]]

        for k in range(len(reference_list)):
            check = []
            for f in reference_metric:
                check.append(reference_list[k] in full[f])
            if all(check):
                intersected_dict[metrics[metric]].append(sorted_dict[metrics[metric]][k])

    return intersected_dict

def extract_intersections(intersect:set,file_list:dict):

    intersect_dict = {keys: [] for keys in file_list}
    for key,items in file_list.items():
        for item in items:
            if datetime.strptime(item.split('_')[-1][:-11], '%Y%m%d') in intersect:
                intersect_dict[key].append(item)

    return intersect_dict

class sortdata():

    def __init__(self,path,metrics,dir):
        
        self.path = path
        self.metrics = metrics
        self.dir = dir

    def __call__(self):
        
        # get file list
        #self.files = import_data_list(self.path)

        # rearrange files according to metric
        self.metric_dict = sort_metric(self.path,self.metrics,self.dir)

        # get the dates which intersect
    
        self.intersect_dict = intersect_date(self.metric_dict)
        
        # sort the files in ascending date order
        for key,items in self.intersect_dict.items():
            self.intersect_dict[key] = sort_files_by_date(items)
        
        return self.intersect_dict

path = r"D:\Swindon_geotiffs"

list_all = import_data_list(path)

metrics = ['VH_COHERENCE','VH','VV_COHERENCE','VV']

