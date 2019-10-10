import pickle
import numpy as np

from abc import ABC, abstractmethod
from tqdm import tqdm, trange
from multiprocessing import Pool
from feature import moment, sift

MOMENT = 0
SIFT = 1

class Compare(object):
    @abstractmethod
    def compare_one(self, rec):
        pass

    def compare_many(self, recs, size):
        if size == 0:
            print("Warning: Comparision with 0 elements")
            return []

        res = []
        p = Pool(processes=10)
        pbar = tqdm(total=size)

        for r in p.imap_unordered(self.compare_one, recs, chunksize=100):
            res.append(r)
            pbar.update()

        res = np.array(res, dtype=[('x', object), ('y', float)])
        res.sort(order="y")

        return res


class CompareMoment(Compare):
    def __init__(self, **kwargs):
        self.key_feats = moment.process_img(str(kwargs["img_path"].resolve()), kwargs["win_h"], kwargs["win_w"])
        self.k = self.key_feats["moments"]
        self.w = np.array(kwargs["weights"])

    def compare_one(self, rec):
        """
        Compares the Color Moments of two images.

        Args:
            rec: The Color Moments to be compared to the Color Moments of the image this class instance was created with.

        Returns:
            The average of the Manhattan distance of the Color Moments across all the windows the image was split into.
        """
        m = pickle.loads(rec["moments"])
        d_m = np.absolute(self.k - m) * self.w

        div = d_m.shape[0]

        res = (
            rec["path"],
            d_m.flatten().sum() / div,
        )

        return res


class CompareSift(Compare):
    def __init__(self, **kwargs):
        self.img_path = kwargs["img_path"]
        self.img_data = sift.process_img(str(self.img_path.resolve()), kwargs["use_opencv"])

    def find_nearest_kps(self, kps, kp):
        """
        Finds the two nearest keypoints from kps to the keypoint kp. This function implements the ratio check to eliminate ambigious matches.

        Args:
            kps: List of keypoints in which to find a match.
            kp: The keypoint who's neighbours are required.

        Returns:
            True if there is a suitable keypoint matching kp, False otherwise.
        """
        best_two = np.sort(np.sum(np.power(kps - kp, 2), axis=1))[:2]
        return 10 * 10 * best_two[0] < 8 * 8 * best_two[1]

    def compare_one(self, img1):
        """
        Compares self.img_data to img1. This function find the number of matching keypoints from self.img_data to img1.

        Args:
            img1: The SIFT features of the image to be compared to.

        Returns:
            A tuple of the image path of img1 and the number of matches.
        """
        img1['sift'] = pickle.loads(img1['sift'])
        res = [
            1 for i in range(0, len(self.img_data['sift'][1]))
            if self.find_nearest_kps(img1['sift'][1], self.img_data['sift'][1]
                                     [i]) == True
        ]

        return (img1['path'], sum(res))

    def compare_many(self, recs, size):
        return np.flip(super().compare_many(recs, size))

comparators = [CompareMoment, CompareSift]

def comparision(t, **opts):
    return comparators[t](**opts)
