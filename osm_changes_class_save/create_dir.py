""" Creates a full path for the files"""
import os

# creates full path
def full_path(folder):
    full_path = os.path.join(os.getcwd(),folder)
    
    return full_path

# creates list of files
def list_file(fullpath):
    list = []
    for files in os.listdir(fullpath):
        if files.endswith('.tiff'):
            list.append(os.path.join(fullpath,files))

    return list

def list_file_unfiltered(fullpath):
    list = []
    for root, _, files in os.walk(fullpath):
        if "To" in root:
            for file in files:
                if file.endswith('.tiff'):
                    list.append(os.path.join(root,file))
    
    return list





