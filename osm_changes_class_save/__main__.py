from osm_changes_class_save.save_class import classsaver
import sys

def main(path,threshold):

    type = ['change','no_change']

    for i in type:

        classsaver(path,threshold,i)  

if __name__ == "__main__":

    path = sys.argv[1]
    threshold = sys.argv[2]
    threshold = int(threshold)

    main(path,threshold)



    

