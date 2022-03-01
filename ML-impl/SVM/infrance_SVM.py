import cv2
import os 
import glob
import joblib
import numpy as np 
import matplotlib.pyplot as plt 
from skimage.feature import hog
from skimage.transform import pyramid_gaussian
from imutils.object_detection import non_max_suppression

min_wdw_sz = (64, 128)
step_size = (10, 10)
downscale = 1.6
orientations = 9
threshold = .90
pixels_per_cell = [6,6]
cells_per_block = [2, 2]

def sliding_window(image, window_size, step_size):
    for y in range(0, image.shape[0], step_size[1]):
        for x in range(0, image.shape[1], step_size[0]):
            yield (x, y, image[y: y + window_size[1], x: x + window_size[0]])

def detector(filename):
    im = cv2.imread(filename)
    im = cv2.resize(im, (200,200))
    clf = joblib.load(os.path.join('./models/3_model.npy'))

    detections = []
    scale = 0

    for im_scaled in pyramid_gaussian(im, downscale = downscale):
        #The list contains detections at the current scale
        if im_scaled.shape[0] < min_wdw_sz[1] or im_scaled.shape[1] < min_wdw_sz[0]:
            break
        for (x, y, im_window) in sliding_window(im_scaled, min_wdw_sz, step_size):
            if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0]:
                continue
            im_window = np.float32(im_window)
            fd =  hog(im_window, orientations=orientations, pixels_per_cell=pixels_per_cell,cells_per_block=cells_per_block,block_norm='L2', transform_sqrt=False,feature_vector=True)
            fd = fd.reshape(1, -1)
            # print(fd.shape)
            pred = clf.predict(fd)

            if pred == 1:
                if clf.decision_function(fd) > threshold:
                    detections.append((int(x * (downscale**scale)), int(y * (downscale**scale)), clf.decision_function(fd), int(min_wdw_sz[0] * (downscale**scale)),int(min_wdw_sz[1] * (downscale**scale))))
                 
        scale += 1
    clone = im.copy()

    rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])
    sc = [score[0] for (x, y, score, w, h) in detections]
    sc = np.array(sc)
    pick = non_max_suppression(rects, probs = sc, overlapThresh = 0.1)
   
    for (x_tl, y_tl, _, w, h) in detections:
        cv2.rectangle(im, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 255, 0), thickness = 2)
    for(xA, yA, xB, yB) in pick:
        cv2.rectangle(clone, (xA, yA), (xB, yB), (0, 255, 0), 2)

    plt.axis("off")
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    plt.title("Raw Detection before NMS")
    plt.show()

    plt.axis("off")
    plt.imshow(cv2.cvtColor(clone, cv2.COLOR_BGR2RGB))
    plt.title("Final Detections after applying NMS")
    plt.show()


def test_folder(foldername):

    filenames = glob.glob(os.path.join(foldername, '*'))
    for filename in filenames:
        detector(filename)


if __name__ == '__main__':
    foldername = './data/test_data'
    test_folder(foldername)