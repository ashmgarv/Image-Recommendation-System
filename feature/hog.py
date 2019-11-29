import cv2
from skimage.feature import hog
import numpy as np
from dynaconf import settings
import pickle


def process_img(img_path):
    scale_factor = settings.HOG.SCALE_FACTOR
    image = cv2.imread(str(img_path),cv2.IMREAD_UNCHANGED)
    width = int(image.shape[1] * scale_factor / 100)
    height = int(image.shape[0] * scale_factor / 100)
    new_res = (width,height)
    resized = cv2.resize(image,new_res,interpolation = cv2.INTER_AREA)
    cell_size = (8,8)
    block_size = (2,2)
    num_bins = 9
    hog_features = hog(resized,orientations=9,pixels_per_cell=cell_size, cells_per_block=block_size, 
                visualize=False, block_norm='L2-Hys',feature_vector=True)
    return {
        'path' : str(img_path),
        'hog' : hog_features
    }

def get_all_vectors(coll, f={}):
    all_image_names = []
    all_vectors = []
    for row in coll.aggregate([{'$match': f}, {'$sort': {'path': 1}}], allowDiskUse=True):
        all_image_names.append(row['path'])
        hog = pickle.loads(row['hog']).flatten()
        all_vectors.append(hog)

    return all_image_names, np.array(all_vectors)