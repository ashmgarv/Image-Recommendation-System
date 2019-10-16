import random
import os
from tasks import task5
import argparse
from pymongo import MongoClient
from dynaconf import settings

def prepare_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', type=str)
    return parser


def run_task_5(data_path, file_path):
    client = MongoClient(host=settings.HOST,
                         port=settings.PORT,
                         username=settings.USERNAME,
                         password=settings.PASSWORD)
    #Run 150 times
    for i in range(150):
        image_name = get_image_name(random.randrange(2,502))
        label = get_label(random.randrange(0,7))
        model = get_model(random.randrange(0,3))
        reduction_technique = get_reduction_technique(random.randrange(0,3))

        img_path = data_path +  '/'  + image_name
        img = client.db['metadata'].find_one({'path':img_path})

        #If the randomly generated image_id does not exist in metadata, then loop again
        if not img:
            continue
        os.system(f"python {file_path} -m {model} -k '4' -frt {reduction_technique} -l {label} -i {image_name} -d {data_path}")

def get_label(num):
    states = [
        'left','right','palmar','dorsal','with_acs','without_acs','male','female'
    ]
    return states[num]

def get_model(num):
    models = [
        'lbp','sift','moment','hog'
    ]
    return models[num]

def get_reduction_technique(num):
    techniques = [
        'pca','nmf','svd','lda'
    ]
    return techniques[num]

def get_image_name(image_num):
    num_zeroes = 7 - len(str(image_num)) % 7
    img_zeroes = ""
    for k in range(num_zeroes):
        img_zeroes += '0'
    img_name = 'Hand_' + str(img_zeroes) + str(image_num) + '.jpg'
    return img_name

if __name__ == "__main__":
    parser = prepare_parser()
    args = parser.parse_args()
    run_task_5(args.data_path, os.getcwd() + '/tasks/task5.py')


