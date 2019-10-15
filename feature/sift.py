import timeit
import cv2
import numpy as np

from dynaconf import settings
from pymongo import MongoClient, UpdateOne
from bson import Binary
from sklearn.cluster import MiniBatchKMeans

import copyreg
import pickle
import sys
from tqdm import tqdm
from functools import partial
import multiprocessing as mp


def talk(args, path, stdout=True, stdin=False, dry_run=False):
    """
    Execute a process with a command.
    Args:
        args: Command to run
        path: Path to run the command in
        stdout: Capture the STDOUT and return
        stdin: Send input to the command
        dry_run: Don't execute the command

    Returns:
        Returns a tuple of (return code, the output of the command in STDOUT and the output of STDERR,)
    """
    # print("Running command: {}".format(" ".join(args)))
    if dry_run:
        return 0, None

    p = Popen(args,
              cwd=path,
              stdout=None if stdout == False else PIPE,
              stdin=None if stdin == False else PIPE,
              stderr=PIPE)
    if stdin:
        comm = p.communicate(stdin)
    elif stdout:
        comm = p.communicate()
    else:
        return (p.returncode, None, None)

    out, err = None if comm[0] == None else comm[0].decode(
        "utf-8"), None if comm[1] == None else comm[1].decode("utf-8")
    return (
        p.returncode,
        out,
        err,
    )


# Taken from https://stackoverflow.com/a/48832618
# Adds support to pickle cv2.KeyPoint objects
def _pickle_keypoints(point):
    return cv2.KeyPoint, (*point.pt, point.size, point.angle, point.response,point.octave, point.class_id)
copyreg.pickle(cv2.KeyPoint().__class__, _pickle_keypoints)


def parse_sift_op(out):
    """
    Parses the sift output from the SIFT binary provided by David Lowe.

    Args:
        out: The output of the SIFT binary

    Returns:
        An array of cv2.Keypoint and list of descriptors associated with each keypoint.
    """
    data = out.split("\n")
    keypoints, var = (int(i) for i in data[0].split(" "))
    data = data[1:]

    tmp = []
    keys = []
    desc = []
    for line in data:
        if len(line.strip()) == 0:
            continue
        if not line.startswith(" "):
            if len(tmp) == var + 4:
                keys.append(cv2.KeyPoint(tmp[1], tmp[0], tmp[2], tmp[3]))
                desc.append(tmp[4:])
            elif len(keys) != 0:
                raise Exception("Expected {} sized vector, got {}".format(
                    var + 4, len(tmp)))
            tmp = []
        tmp.extend(
            [float(i) if '.' in i else int(i) for i in line.strip().split(" ")])

    if len(tmp) > 0:
        keys.append(cv2.KeyPoint(tmp[1], tmp[0], tmp[2], tmp[3]))
        # keys.append((tmp[1], tmp[0], tmp[2], tmp[3],))
        desc.append(tmp[4:])

    if len(keys) != keypoints:
        raise Exception("Expected {} keypoints, got {}".format(
            keypoints, len(keys)))
    return [keys, np.array(desc)]


def img_sift(img_path, sift_opencv):
    """
    Extracts the SIFT keypoints from a provided image.

    Args:
        img_path: The given image to operate on.
        sift_opencv: An instance of OpenCV's sift. If not provided, the application will fallback to the SIFT binary by David Lowe.

    Returns:
        An array of cv2.KeyPoint and list of descriptors associated with each keypoint.
    """
    # TODO: Resize image?
    # sift binary does not ally images having > 1800 pixels in any dimension.
    # Also we have to maintain aspect ratio.

    # img = cv2.imread(str(img_path))
    # img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

    if sift_opencv:
        # sift = cv2.xfeatures2d.SIFT_create()
        # res = sift.detectAndCompute(img_gray, None)
        res = sift_opencv.detectAndCompute(img_gray, None)
        # cv2.KeyPoint cannot be pickled. So multiprocessing with this is not
        # possible
        # res = [[(i.pt[0], i.pt[1], i.size, i.angle,) for i in res[0]], res[1]]
        res = [res[0], res[1]]
        return res

    img_data = cv2.imencode(".pgm", img_gray)[1].tostring()

    ret, out, err = talk([settings.SIFT.BIN_PATH],
                               settings._root_path,
                               stdin=img_data,
                               stdout=True)
    if err.startswith("Finding keypoints") and err.endswith(
            "keypoints found.\n"):
        err = None

    if ret != 0 or err != None:
        raise Exception("Error occured: {}".format(err))

    try:
        return parse_sift_op(out)
    except Exception as e:
        raise Exception("Invalid output from sift binary: {}\n{}".format(
            e, out))


def process_img(img_path, use_opencv):
    return {
        "path":
            str(img_path),
        "sift":
            img_sift(str(img_path),
                     cv2.xfeatures2d.SIFT_create() if use_opencv else None)
    }


def visualize_sift(img_path, op_path):
    """
    Draws the keypoints on the image and writes the image on the disk.

    Args:
        img_path: The image who's SIFT features are to be visualized.
        op_path: Path to write the output image.
    """
    img = cv2.imread(str(img_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detectAndCompute(gray, None)
    img = cv2.drawKeypoints(gray,
                            kp[0],
                            img,
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(
        str(op_path / '{}_keypoints.jpg'.format(img_path.resolve().name)), img)


#generates a dictionary of image_path to the subject id. {'Hand_x': 1, 'Hand_y':1, 'Hand_z':2.....}
def get_path_to_subject_map(image_paths):
    client = MongoClient(host=settings.HOST,
                         port=settings.PORT,
                         username=settings.USERNAME,
                         password=settings.PASSWORD)
    coll = client.db[settings.IMAGES.METADATA_COLLECTION]

    path_to_subject_map = {}
    for row in coll.find({'path': {'$in': image_paths}}, {'id': 1, 'path': 1}):
        path_to_subject_map[row['path']] = row['id']
    
    return path_to_subject_map


#generates a single keypoint histogram vector for each image
def get_histogram_vector(kmeans_cluster, subject_count, ordered_vector):
    index = ordered_vector[0]
    all_keypoints = ordered_vector[1]

    histogram_vector = np.zeros(subject_count * 10)
    kp_count = all_keypoints.shape[0]

    for key_point in all_keypoints:
        idx = kmeans_cluster.predict(key_point.reshape(1,-1))
        histogram_vector[idx] += 1/kp_count

    return (index, histogram_vector)


def generate_histogram_vectors(coll):
    # Fetch ids, image paths and descriptor arrays of all inserted hog vectors
    print("fetching all keypoints....")
    row_ids, image_paths, all_keypoints = [], [], []
    for row in coll.find({}):
        row_ids.append(row['_id'])
        image_paths.append(row['path'])
        all_keypoints.append(pickle.loads(row['sift'])[1])
    stacked_keypoints = np.vstack(all_keypoints)

    # Subject count is used to classify clusters. Image count is for kmeans batchsize
    print("bucketing into subjects....")
    path_to_subject_map = get_path_to_subject_map(image_paths)
    subject_count = len(set(list(path_to_subject_map.values())))
    image_count = len(image_paths)

    # Build kmeans cluster with all the descriptors
    k = subject_count * 10
    batch_size = image_count * 3
    print("Building {} clusters with batch size {} ....".format(k, batch_size))
    kmeans_cluster = MiniBatchKMeans(n_clusters=k, batch_size=batch_size, verbose=1).fit(stacked_keypoints)

    # Index each vector to preserve order after multiprocessing -> [(1,v1), (2,v2), (3,v3), .....]
    ordered_vectors = [(i, all_keypoints[i]) for i in range(len(all_keypoints))]

    # Run with multiprocessing to get single vector from multiple keypoints for all images.
    # Upsert using index from ordered_vectors
    print("generating histogram vectors for every image....")
    pool = mp.Pool(processes=mp.cpu_count())
    partial_get_histogram_vector = partial(get_histogram_vector, kmeans_cluster, subject_count)
    upserts = []
    for res in tqdm(pool.imap_unordered(partial_get_histogram_vector, ordered_vectors), total=len(ordered_vectors)):
        index = res[0]
        single_vector = Binary(pickle.dumps(res[1], protocol=2))
        upserts.append(
            UpdateOne(
                filter = {'_id': row_ids[index]},
                update = {'$set': {'_id': row_ids[index], 'histogram_vector': single_vector}},
                upsert = True
            )
        )
        if len(upserts) == settings.LOADER.BATCH_SIZE:
            coll.bulk_write(upserts)
            upserts.clear()

    if upserts:
        coll.bulk_write(upserts)
    print("fin.")

def get_all_vectors(coll, f={}):
    all_image_names = []
    all_vectors = []
    for row in coll.find(f):
        all_image_names.append(row['path'])
        histogram_vectors = pickle.loads(row['histogram_vector'])
        all_vectors.append(histogram_vectors)

    return all_image_names, np.array(all_vectors)
