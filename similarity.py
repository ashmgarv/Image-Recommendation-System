import moment
import cv2
import numpy as np
from pymongo import MongoClient

from dynaconf import settings

def calc_mom_sim(img_path, k):
    client = MongoClient(host=settings.HOST, port=settings.PORT, username=settings.USERNAME, password=settings.PASSWORD)
    coll = client.db[settings.MOMENT.COLLECTION]

    key_feats = moment.process_img(img_path, settings.WINDOW.WIN_HEIGHT, settings.WINDOW.WIN_WIDTH)
    k_y = np.array(key_feats["y_moments"], dtype=list)
    k_u = np.array(key_feats["u_moments"], dtype=list)
    k_v = np.array(key_feats["v_moments"], dtype=list)

    def image_index(rec):
        y = np.array(rec["y_moments"], dtype=list)
        u = np.array(rec["u_moments"], dtype=list)
        v = np.array(rec["v_moments"], dtype=list)

        tmp = np.absolute(k_y - y)
        tmp[:,0] *= settings.MOMENT.W_Y_1
        tmp[:,1] *= settings.MOMENT.W_Y_2
        tmp[:,2] *= settings.MOMENT.W_Y_3
        d_y = np.sum(tmp, axis=1)

        tmp = np.absolute(k_u - u)
        tmp[:,0] *= settings.MOMENT.W_U_1
        tmp[:,1] *= settings.MOMENT.W_U_2
        tmp[:,2] *= settings.MOMENT.W_U_3
        d_u = np.sum(tmp, axis=1)

        tmp = np.absolute(k_v - v)
        tmp[:,0] *= settings.MOMENT.W_V_1
        tmp[:,1] *= settings.MOMENT.W_V_2
        tmp[:,2] *= settings.MOMENT.W_V_3
        d_v = np.sum(tmp, axis=1)

        d = d_y + d_u + d_v
        return (rec["path"], d.mean(),)

    data = coll.find()
    d = np.array([image_index(data.next()) for _ in range(0, coll.count_documents({}))], dtype=[('x', object), ('y', float)])
    d.sort(order="y")

    return d[0:k]
