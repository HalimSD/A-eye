import glob
import os
import joblib
import cv2
import shutil
import numpy as np
from skimage.transform import pyramid_gaussian
from skimage import color
from skimage.feature import hog
import matplotlib.pyplot as plt 
from imutils.object_detection import non_max_suppression

dest_folder = './data/test_data'

if os.path.isdir(dest_folder) == False:
    os.mkdir(dest_folder)
    for folder in os.listdir('./data'):
        if folder == '0':
            for file in os.listdir('./data/0')[:5]:
                try:
                    to_be_copied = os.path.join('./data/0',file)
                    shutil.copy(to_be_copied, dest_folder)
                except Exception as e:
                    print(str(e))
        if folder == '1':
            for file in os.listdir('./data/1')[:5]:
                try:
                    to_be_copied = os.path.join('./data/1',file)
                    shutil.copy(to_be_copied, dest_folder)
                except Exception as e:
                    print(str(e))


model_path = joblib.load(os.path.join('./models/3_model.npy'))
detections = []
min_wdw_sz = (64, 128)
step_size = (10, 10)
downscale=1.6


def sliding_window(image, window_size, step_size):
    for y in range(0, image.shape[0], step_size[1]):
        for x in range(0, image.shape[1], step_size[0]):
            yield (x, y, image[y: y + window_size[1], x: x + window_size[0]])

def detect(test_file):
    img = cv2.imread(test_file,  cv2.IMREAD_GRAYSCALE)
    resized_image = cv2.resize(img, (64,128))
    scale = 0

    for im_scaled in pyramid_gaussian(resized_image, downscale):
        if im_scaled.shape[0] < min_wdw_sz[1] or im_scaled.shape[1] < min_wdw_sz[0]:
            break

        for (x, y, im_window) in sliding_window(im_scaled, min_wdw_sz, step_size):
            if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0]:
                continue
            # im_window = color.rgb2gray(im_window)
            fd =  hog(im_window, orientations=9, pixels_per_cell=(9,9),cells_per_block=(2, 2),block_norm='L2',transform_sqrt=False,feature_vector=True,visualize=False)
            fd = fd.reshape(1, -1)
            pred = model_path.predict(fd)
            print(pred)

            if pred == 1:                
                if model_path.decision_function(fd) > 0.5:
                    detections.append((int(x * (downscale**scale)), int(y * (downscale**scale)), model_path.decision_function(fd), int(min_wdw_sz[0] * (downscale**scale)),int(min_wdw_sz[1] * (downscale**scale))))                   
        scale += 1
    clone = resized_image.copy()

    rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])
    sc = [score[0] for (x, y, score, w, h) in detections]
    # print ("sc: ", sc)
    sc = np.array(sc)
    pick = non_max_suppression(rects, probs = sc, overlapThresh = 0.3)

    for (x_tl, y_tl, _, w, h) in detections:
        cv2.rectangle(resized_image, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 255, 0), thickness = 2)
    for(xA, yA, xB, yB) in pick:
        cv2.rectangle(clone, (xA, yA), (xB, yB), (0, 255, 0), 2)

    plt.axis("off")
    plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
    plt.title("Raw Detection before NMS")
    plt.show()

    plt.axis("off")
    plt.imshow(cv2.cvtColor(clone, cv2.COLOR_BGR2RGB))
    plt.title("Final Detections after applying NMS")
    plt.show()
    
def test_folder(foldername):
    filenames = glob.glob(os.path.join(foldername, '*'))
    for filename in filenames:
        detect(filename)

if __name__ == '__main__':
    foldername = './data/test_data'
    test_folder(foldername)
