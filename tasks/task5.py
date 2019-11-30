import argparse
import sys
from dynaconf import settings
import numpy as np
import time
from sklearn.metrics.pairwise import euclidean_distances
from pathlib import Path

sys.path.append('../')
import output
import scipy.sparse
from utils import get_all_vectors, store_output

def prepare_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--layers', type=int, required=True)
    parser.add_argument('-k', '--hashes', type=int, required=True)
    return parser

def get_hash(all_planes, vector):
    layer_hash = all_planes.dot(vector.transpose())
    list_of_hashes = ["1" if i>0 else "0" for i in layer_hash]
    hash = "".join(list_of_hashes)
    return hash

def get_euclidean_distance(vec1, vec2):
    return euclidean_distances(vec1.reshape(1,-1), vec2.reshape(1,-1))

def get_nearest_hash(key, k_list):
    distances = {}
    main_arr = [float(c) for c in key]
    for k in k_list:
        hamming_distance = scipy.spatial.distance.hamming(main_arr, [float(c) for c in k])
        distances[k] = hamming_distance
    return sorted(distances.items(), key=lambda item: item[1])


def get_nearest_members(query, layers, planes_per_layer, retreived_keys):
    members = []
    for i, layer in enumerate(layers):
        key = get_hash(planes_per_layer[i], query)
        all_keys = list(layer.keys())

        if key in all_keys:
            all_keys.remove(key)
        more_keys = [ks for ks in all_keys if ks not in retreived_keys]

        #retrieve closet key
        closet_key = get_nearest_hash(key, more_keys)[0][0]
        members += layer[closet_key]
        retreived_keys.append(closet_key)
    return members, retreived_keys


#To perform the lsh indexing
def perform_lsh(inp_index, input_vector, layers, planes_per_layer, images):
    #Iterate over each layer
    for i, layer in enumerate(layers):
        key = get_hash(planes_per_layer[i], input_vector)

        if key not in layer:
            layer[key] = []
        layer[key].append(images[inp_index])


def query_relevant_images(query_vec, t, layers, planes_per_layer,data_matrix,images):
    members = []
    final_dictionary = {}
    retreived_keys = []

    for index in range(len(layers)):
        key = get_hash(planes_per_layer[index], query_vec)
        if key in layers[index]:
            members += layers[index][key]
            retreived_keys.append(key)

    member_count = len(members)
    members = list(set(members))
    unique_member_count = len(members)

    
    while unique_member_count < t:
        closet_members, retreived_keys = get_nearest_members(query_vec, layers, planes_per_layer, retreived_keys)
        members += closet_members

        #Count the number of candidate members
        member_count = len(members)
        members = list(set(members))
        unique_member_count = len(members)

    for cur_member in members:
        idx  = images.index(cur_member)
        member_vector = data_matrix[idx]
        distance = get_euclidean_distance(member_vector, query_vec)
        final_dictionary[cur_member] = distance


    final_sorted = sorted(final_dictionary.items(), key=lambda item: item[1])
    return final_sorted[:t], member_count, unique_member_count


if __name__ == "__main__":
    parser = prepare_parser()
    args = parser.parse_args()

    start = time.time()

    #Part a
    l = args.layers
    k = args.hashes

    #load the object-feature matrix and data 
    images, data_matrix = get_all_vectors('moment', master_db=True)
    
    data_matrix_shape = data_matrix.shape[1]

    layers = [{} for _ in range(l)]

    planes_per_layer = []

    for i in range(l):
        #Generate normally distributed planes
        planes = np.random.randn(k, data_matrix_shape)

        #Generating compressed sparsed row matrices
        planes_per_layer.append(scipy.sparse.csr_matrix(planes))
    
    #index all points
    for i in range(data_matrix.shape[0]):
        perform_lsh(i, data_matrix[i], layers, planes_per_layer,images)

    print ("\nIndex structure created.")
    print ("\nTime Taken: ", (time.time()-start))

    #Part b
    query = input("\nEnter the query image id:\n")
    to_output = query.split('.')[0]
    t = int(input("Enter t:\n"))

    data_path = Path(settings.path_for(settings.MASTER_DATA_PATH)).resolve()

    query = str(data_path / query)

    index = images.index(query)
    query_vec = data_matrix[index]

    results = query_relevant_images(query_vec, t, layers, planes_per_layer,data_matrix,images)

    images = results[0]
    member_count = results[1]
    unique_member_count = results[2]

    image_paths = []
    all_images = []
    for img in images:
        image_paths.append([img[0].split('/')[-1],img[0]])
        all_images.append(img[0])

    print ("Total no. of unique images considered: ", unique_member_count)
    print ("Total no. of overall images: ", member_count)

    #Store output for task 5
    store_output(all_images)

    output.write_to_file("task5.html",
                         f"task5-{to_output}.html",
                         key=query,
                         items=image_paths,
                         title="Task5")
    
