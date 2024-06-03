""" Gets for a label, any training tile which intersects with it """

# Function which checks if A (label) intersects with B (training SAR data)

def check_intersection(train: tuple,label: tuple):
    """Takes in coords from training SAR data and label data and returns True if intersects else False"""
    # Extract data
    train_min_long, train_min_lat, train_max_long, train_max_lat = train[0], train[1], train[2], train[3]
    label_min_long, label_min_lat, label_max_long, label_max_lat = label[0], label[1], label[2], label[3]

    min_long_check = train_max_long <= label_min_long
    max_long_check = train_min_long >= label_max_long
    if min_long_check or max_long_check:
        return False
    
    min_lat_check = train_max_lat <= label_min_lat
    max_lat_check = train_min_lat >= label_max_lat
    if min_lat_check or max_lat_check:
        return False
    else:
        return True

class overlap_training_SAR_data():
    """ Creates an instance which has memory of the training data that overlaps with a certain label """
    def __init__(self,label_cooords:tuple,list_of_SAR_coords:list):

        self.label_coords = label_cooords
        self.list_of_SAR_coords = list_of_SAR_coords

    def get_training_data(self):

        self.training_data = []
        self.training_data_path = []
        
        for data in self.list_of_SAR_coords:
            # for each available training data, check if it overlaps with the label
            result = check_intersection(data, self.label_coords)
            if result:
                self.training_data.append(data)
                self.training_data_path.append(data[-1])