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
    mean = window.flatten().mean()
    temp = np.fromfunction(lambda x,y: (window[x,y] - mean) ** 3, window.shape, dtype=int).flatten().sum() / (window.shape[0] * window.shape[1])
    if temp >= 0:
        return temp ** 1/3
    temp *= -1
    return (temp ** 1/3) * -1

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

            win = y[i:i+win_h, j:j+win_w]
            y_mom_feat.append([moment_1(win), moment_2(win), moment_3(win)])

            win = u[i:i+win_h, j:j+win_w]
            u_mom_feat.append([moment_1(win), moment_2(win), moment_3(win)])

            win = v[i:i+win_h, j:j+win_w]
            v_mom_feat.append([moment_1(win), moment_2(win), moment_3(win)])

    return y_mom_feat, u_mom_feat, v_mom_feat
