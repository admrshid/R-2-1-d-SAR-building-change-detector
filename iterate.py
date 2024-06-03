import os
import json
import sys

def main(layer,output,coord):

    path = os.path.join(os.getcwd(),'osm_changes','config','default.json')

    with open(path, 'r') as file:
        data = json.load(file)

    # Update the values
    data['layer1'] = layer[0]
    data['layer2'] = layer[1]
    data['output_dir'] = './' + output
    data['min_latitude'] = coord[0]
    data['max_latitude'] = coord[1]
    data['min_longitude'] = coord[2]
    data['max_longitude'] = coord[3]

    # Open the JSON file again in write mode to update it
    with open(path, 'w') as file:
        json.dump(data, file, indent=4)

if __name__ == '__main__':
    # Check if the correct number of arguments is provided
    
    #if len(sys.argv) != 3:
        #print("Usage: python script.py layer1 layer2 output_dir")
        #sys.exit(1)

    # Extract the command-line arguments
    
    layer = [sys.argv[1], sys.argv[2]]
    output = sys.argv[3]
    coord = [sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7]]
    for i in range(len(coord)):
        coord[i] = float(coord[i])

    # Call the main function
    main(layer, output, coord)


