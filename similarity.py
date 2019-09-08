import moment
import cv2
import numpy as np
from pymongo import MongoClient

WIN_H = 100
WIN_W = 100

W_Y_1 = 1
W_U_1 = 1
W_V_1 = 1

W_Y_2 = 1
W_U_2 = 1
W_V_2 = 1

W_Y_3 = 1
W_U_3 = 1
W_V_3 = 1

def process_img(img_path):
    img = cv2.imread(str(img_path))
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    y, u, v = moment.img_moment(img_yuv, WIN_H, WIN_W)
    return {
        "path": str(img_path),
        "y_moments": y,
        "u_moments": u,
        "v_moments": v
    }


def calculate_similarity(self, coll_name, img_path, k):
    client = MongoClient('mongodb://localhost:27017/')
    coll = client.db[coll_name]

    key_feats = process_img(img_path)
    k_y = np.array(key_feats["y_moments"], dtype=list)
    k_u = np.array(key_feats["u_moments"], dtype=list)
    k_v = np.array(key_feats["v_moments"], dtype=list)

    def image_index(rec):
        y = np.array(rec["y_moments"], dtype=list)
        u = np.array(rec["u_moments"], dtype=list)
        v = np.array(rec["v_moments"], dtype=list)

        tmp = np.absolute(k_y - y)
        tmp[:,0] *= W_Y_1
        tmp[:,1] *= W_Y_2
        tmp[:,2] *= W_Y_3
        d_y = np.sum(tmp, axis=1)

        tmp = np.absolute(k_u - u)
        tmp[:,0] *= W_U_1
        tmp[:,1] *= W_U_2
        tmp[:,2] *= W_U_3
        d_u = np.sum(tmp, axis=1)

        tmp = np.absolute(k_v - v)
        tmp[:,0] *= W_V_1
        tmp[:,1] *= W_V_2
        tmp[:,2] *= W_V_3
        d_v = np.sum(tmp, axis=1)

        d = d_y + d_u + d_v
        return (rec["path"], d.mean(),)

    data = coll.find()
    d = np.array([image_index(data.next()) for _ in range(0, coll.count_documents({}))], dtype=[('x', object), ('y', float)])
    d.sort(order="y")

    return d[0:k]
