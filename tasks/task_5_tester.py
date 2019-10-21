import random
import os
from tasks import task5
import argparse
from pymongo import MongoClient
from dynaconf import settings
from pathlib import Path

def prepare_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', type=str)
    return parser


def run_task_5(data_path, file_path):
    client = MongoClient(host=settings.HOST,
                         port=settings.PORT,
                         username=settings.USERNAME,
                         password=settings.PASSWORD)

    data_path = Path(settings.path_for(data_path))
    paths = list(data_path.iterdir())

    positive_res = 0
    negative_res = 0

    for path in paths:
        path = str(path)
        image_name = path.split('/')[-1]
        label = get_label(random.randrange(0,7))
        model = get_model(random.randrange(0,3))
        reduction_technique = get_reduction_technique(random.randrange(0,3))

        img_path = path
        img = client.db['metadata'].find_one({'path':img_path})

        #If the randomly generated image_id does not exist in metadata, then loop again
        if not img:
            continue

        original_label = ""
        if label in ('male', 'female'):
            original_label = img['gender']
        elif label in ('palmar', 'dorsal'):
            original_label = img['aspectOfHand'].split(' ')[0]
        elif label in ('left', 'right'):
            original_label = img['aspectOfHand'].split(' ')[1]
        elif label in ('with_acs', 'without_acs'):
            original_label = img['accessories']

        print("Given label : " +label)
        print('Original label of the image is : ' + original_label)

        res = task5.run(model, 30, reduction_technique, label, image_name, data_path)

        if original_label == res:
            positive_res += 1
        else:
            negative_res += 1

    print("We got the accuracy of " + str((positive_res / (positive_res + negative_res)) * 100))


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


