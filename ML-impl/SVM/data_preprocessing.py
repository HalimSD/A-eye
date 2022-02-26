# Data from https://www.kaggle.com/constantinwerner/human-detection-dataset 
# HOG feature extractor: 1- preprocess imgs to 1:2 scale. 2- calculate gradients. 3- calculate magnitudes & orientation
import cv2
import os
import numpy as np
from skimage.io import imread
from skimage.feature import hog
from sklearn.externals import joblib
import glob

data_src = './data'
path_pos = './data/1'
path_neg = './data/0'
path_pos_features = './data/features/pos'
path_neg_features = './data/features/neg'

positive_images = [os.path.join(path_pos, file) for file in os.listdir(path_pos) if '.png' in file]
negative_images = [os.path.join(path_neg, file) for file in os.listdir(path_neg) if '.png' in file]


def resize_images():
    for folder in os.listdir(data_src):
        if "." not in folder:
            for image in os.listdir(os.path.join(data_src,folder)):
                if ".png" in image:
                    try:
                        image_path = os.path.join(data_src, os.path.join(folder, image))
                        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                        resized_image = cv2.resize(img, (64,128))
                        cv2.imwrite(image_path,resized_image)
                    except Exception as e:
                        print(str(e))
                else: 
                    print(os.path.join(folder, image))


def extract_features():
    if not os.path.isdir(path_pos_features):
        os.makedirs(path_pos_features)

    if not os.path.isdir(path_neg_features):
        os.makedirs(path_neg_features)

    for im_path in glob.glob(os.path.join(path_pos, "*")):
        image = imread(im_path)
        fd =hog(image, orientations=9, pixels_per_cell=(9, 9), cells_per_block=(2, 2), block_norm='L1', visualize=False, transform_sqrt=False, feature_vector=True)
        fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
        fd_path = os.path.join(path_pos_features, fd_name)
        joblib.dump(fd, fd_path)
    print("Positive features saved in {}".format(path_pos_features))

    for im_path in glob.glob(os.path.join(path_neg, "*")):
        image = imread(im_path)
        fd =hog(image, orientations=9, pixels_per_cell=(9, 9), cells_per_block=(2, 2), block_norm='L1', visualize=False, transform_sqrt=False, feature_vector=True)
        fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
        fd_path = os.path.join(path_neg_features, fd_name)
        joblib.dump(fd, fd_path)
    print("Negative features saved in {}".format(path_neg_features))
    

if __name__=='__main__':
    resize_images()
    extract_features()