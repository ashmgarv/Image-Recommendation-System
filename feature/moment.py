import cv2
import numpy as np

#img_bgr = cv2.imread("/home/amitab/Documents/MWDB Project/Phase 0/mia-khalifa.jpg")
#img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2YUV)
#y,u,v=cv2.split(img_yuv)


def moment_1(window):
    return window.flatten().mean()


def moment_2(window):
    return window.flatten().std()


def moment_3(window):
    std = window.std()
    if std == 0 or window.size == 0:
        return 0
    return np.power(
        (window - window.mean()) / window.std(), 3).sum() / window.size
    # mean = window.flatten().mean()
    # temp = np.fromfunction(lambda x,y: (window[x,y] - mean) ** 3, window.shape, dtype=int).flatten().sum() / (window.shape[0] * window.shape[1])
    # if temp >= 0:
    #     return temp ** 1/3
    # temp *= -1
    # return (temp ** 1/3) * -1


def img_moment(img, win_h, win_w):
    y, u, v = cv2.split(img)
    img_h, img_w, chans = img.shape

    y_mom_feat = []
    u_mom_feat = []
    v_mom_feat = []
    # Ignore the left out pixels? Or perhaps take another window overlapping the
    # last window created.
    for i in range(0, img_h, win_h):
        if i + win_h > img_h:
            break
        for j in range(0, img_w, win_w):
            if j + win_w > img_w:
                break

            win = y[i:i + win_h, j:j + win_w]
            y_mom_feat.append([moment_1(win), moment_2(win), moment_3(win)])

            win = u[i:i + win_h, j:j + win_w]
            u_mom_feat.append([moment_1(win), moment_2(win), moment_3(win)])

            win = v[i:i + win_h, j:j + win_w]
            v_mom_feat.append([moment_1(win), moment_2(win), moment_3(win)])

    return y_mom_feat, u_mom_feat, v_mom_feat


def process_img(img_path, win_h, win_w):
    img = cv2.imread(str(img_path))
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    y, u, v = img_moment(img_yuv, win_h, win_w)
    return {
        "path": str(img_path),
        "y_moments": y,
        "u_moments": u,
        "v_moments": v
    }

def make_lut_u():
    return np.array([[[i,255-i,0] for i in range(256)]],dtype=np.uint8)

def make_lut_v():
    return np.array([[[0,255-i,i] for i in range(256)]],dtype=np.uint8)

# Taken from https://stackoverflow.com/a/43988642
def visualize_yuv(img_path, op_path):
    img = cv2.imread(str(img_path))

    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(img_yuv)

    lut_u, lut_v = make_lut_u(), make_lut_v()

    # Convert back to BGR so we can apply the LUT and stack the images
    y = cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)
    u = cv2.cvtColor(u, cv2.COLOR_GRAY2BGR)
    v = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)

    u_mapped = cv2.LUT(u, lut_u)
    v_mapped = cv2.LUT(v, lut_v)

    result = np.vstack([img, y, u_mapped, v_mapped])

    cv2.imwrite(str(op_path / '{}_yuv.png'.format(img_path.resolve().name)), result)

def visualize_moments(img_path, op_path):
    img = cv2.imread(str(img_path))
    img_h, img_w, chans = img.shape

    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(img_yuv)

    img_y = np.empty(img.shape, dtype=np.uint8)
    img_u = np.empty(img.shape, dtype=np.uint8)
    img_v = np.empty(img.shape, dtype=np.uint8)

    lut_u, lut_v = make_lut_u(), make_lut_v()

    for i in range(0, img_h, 100):
        if i + 100 > img_h:
            break
        for j in range(0, img_w, 100):
            if j + 100 > img_w:
                break

            win_y = y[i:i + 100, j:j + 100]
            win_u = u[i:i + 100, j:j + 100]
            win_v = v[i:i + 100, j:j + 100]
            # y_mom_feat.append([moment_1(win), moment_2(win), moment_3(win)])
            img_y[i:i + 100, j:j + 100] = [moment_1(win_y), moment_2(win_y), moment_3(win_y)]
            img_u[i:i + 100, j:j + 100] = [moment_1(win_u), moment_2(win_u), moment_3(win_u)]
            img_v[i:i + 100, j:j + 100] = [moment_1(win_v), moment_2(win_v), moment_3(win_v)]

    result_y = np.vstack([cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)] + [cv2.cvtColor(i, cv2.COLOR_GRAY2BGR) for i in cv2.split(img_y)])
    result_u = np.vstack([cv2.LUT(cv2.cvtColor(u, cv2.COLOR_GRAY2BGR), lut_u)] + [cv2.LUT(cv2.cvtColor(i, cv2.COLOR_GRAY2BGR), lut_u) for i in cv2.split(img_u)])
    result_v = np.vstack([cv2.LUT(cv2.cvtColor(v, cv2.COLOR_GRAY2BGR), lut_v)] + [cv2.LUT(cv2.cvtColor(i, cv2.COLOR_GRAY2BGR), lut_v) for i in cv2.split(img_v)])
    cv2.imwrite(str(op_path / '{}_y.png'.format(img_path.resolve().name)), result_y)
    cv2.imwrite(str(op_path / '{}_u.png'.format(img_path.resolve().name)), result_u)
    cv2.imwrite(str(op_path / '{}_v.png'.format(img_path.resolve().name)), result_v)
