import os
import fnmatch
import math
import numpy as np
import random
import rasterio as r
from skimage.transform import resize

# data must be saved as class\video\frame.png

path = r"D:\Swindon_geotiffs\video_dataset"

class retrieve_filename:

    # input: dataset file path

    # output: list of filenames

    def __init__(self,path,types,segment):

        self.segment = segment
        self.path = path
        self.types = types

    def __call__(self):
        all_files = []
        for dir,root,file in os.walk(self.path):
            if len(file) == 0 and os.path.basename(dir) in self.types and not self.segment:
                for r in root:
                    all_files.append(os.path.join(dir,r))
            elif len(file) == 0 and fnmatch.fnmatch(os.path.basename(dir),'segment*') and self.segment:
                all_files.append(dir)

        return all_files

def getclass(file,types):
    out = []
    file_str = str(file)
    for ty in types:
        if fnmatch.fnmatch(file_str,"*\\"+ty+"\\*" ):
            out.append(ty)
    if len(out) != 1:
        print(f'{file} has more than one class description in its file name')
    else:
        return out[0]

class arrange_data:

    # rearranges data into classes

    # input: list of files 

    # output: dict of files with keys as class

    def __init__(self,files,types):
        
        self.files = files
        self.types = types

    def __call__(self):

        dict = {}
        for i in self.files:
            classes = getclass(i,self.types)
            filename = i
            if classes not in dict.keys():
                dict[classes] = []
            dict[classes].append(filename)
                
        return dict

def encodeclass(arrange_data):
    dict = {}
    length = len(arrange_data.keys())
    encode = np.arange(0,length,1)
    classes = list(arrange_data.keys())
    for i in range(length):
        if i not in dict:
            dict[classes[i]] = []
        dict[classes[i]] = encode[i]
    return dict

class train_test_split:
    # splits arranged data into train and test

    # input: arranged data, dict of train test split

    # output: dict of train test as keys and (class,filename) tuple as value

    def __init__(self,arranged_data,split):
        
        self.train = split['train']
        self.test = split['test']
        self.classes = [i for i in arranged_data.keys()]
        self.arranged_data = arranged_data

    def __call__(self):

        traindata = []
        testdata = []
        for i in self.classes:
            
            class_type = i
            n_train = math.ceil(self.train*len(self.arranged_data[class_type]))
            n_test = len(self.arranged_data[class_type]) - n_train

            traindata.append((class_type,self.arranged_data[class_type][:n_train]))
            testdata.append((class_type,self.arranged_data[class_type][n_train:n_train+n_test]))

        return traindata, testdata

class npframegenerator:
    def __init__(self,data,encoded_class:dict,num_frames,frame_step,metrics_include,training = False):

        self.data = data # this is from train, test
        self.encoded_class = encoded_class # dict carrying encoded label
        self.training = training
        self.num_frames = num_frames
        self.frame_step = frame_step
        self.metrics_include = metrics_include

    def __call__(self):
        label = []
        target = []
        for i in range(len(self.data)):

            classname = self.data[i][0]
            datas = self.data[i][1]

            for j in datas:

                metrics = os.listdir(j)  # get list of metrics
                check = [m in metrics for m in self.metrics_include]
                if not all(check):
                    print(f'Some metrics to be included does not exist in metrics available')
                    return

                vid_metric = []
                
                for m in self.metrics_include:

                    full_path = os.path.join(j,m) # this is one element ex: 32449_21776_16_filtered\VV
                    vid = os.listdir(full_path) # for each metric get the list of frames 
                    if vid == []:
                        continue

                    fullpath = [os.path.join(full_path,k) for k in vid if k.endswith('tiff')]

                    # calculate required indexes for reduced frame size
                    ending_frame = self.frame_step*(self.num_frames-1)
                    if self.frame_step*(self.num_frames-1) > len(fullpath) - 1:
                        print(f'Ending frame requested: {ending_frame} exceeds available ending frame: {len(fullpath)-1} for {full_path}')
                        return

                    index = [0]
                    for a in range(self.num_frames - 1):
                        index.append(self.frame_step*(a+1) + index[0])
                        
                    numpydata = []
                    for ind in index:
                        s = fullpath[ind]

                        with r.open(s, 'r') as f:
                            img = f.read() # shape = (1,y,x)
                            img = np.squeeze(img)
                            img = resize(img, (30,125), preserve_range=True)
                            img = np.expand_dims(img,axis=0)
                            numpydata.append(img)
                        
                    numpyfull = np.concatenate(numpydata,axis=0) # make into (n,y,x)
                    vid_metric.append(np.expand_dims(numpyfull,axis=3)) # make into (n,y,x,1)

                if vid_metric == []:
                    continue
                vid_full = np.concatenate(vid_metric,axis=3)

                label.append(self.encoded_class[classname])
                target.append(vid_full)

        pairs =  list(zip(target,label))

        if self.training:
            random.shuffle(pairs)
        
        for target, label in pairs:
            yield target, label
 



            



