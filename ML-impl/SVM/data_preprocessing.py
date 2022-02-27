# Data from https://www.kaggle.com/constantinwerner/human-detection-dataset 
# HOG feature extractor: 1- preprocess imgs to 1:2 scale. 2- calculate gradients. 3- calculate magnitudes & orientation
import pickle
import uuid
import cv2
import os
from skimage.feature import hog
import joblib
import numpy as np
from torch import int32

data_src = './data'
path_features = './data/features'

categories = ['0', '1']
data = []
data_features = []

def resize_images():
    for category in categories:

        path = os.path.join(data_src, category)
        label = categories.index(category)

        for image in os.listdir(path):
            if ".png" in image:
                try:
                    image_path = os.path.join(path, image)
                    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    resized_image = cv2.resize(img, (64,128))
                    data.append([resized_image, label])
                except Exception as e:
                    print(str(e))

def extract_features():
    if not os.path.isdir(path_features):
        os.makedirs(path_features)

    for image,label in data:
        fd =hog(image, orientations=9, pixels_per_cell=(9, 9), cells_per_block=(2, 2), block_norm='L1', visualize=False, transform_sqrt=False, feature_vector=True)
        fd_name = str(label) + '-' + uuid.uuid4().hex + ".npy"
        fd_path = os.path.join(path_features, fd_name)
        joblib.dump(fd, fd_path)
        data_features.append([fd,label])

    pick_in = open('./data/picked.pickle', 'wb')
    pickle.dump(data_features, pick_in)
    pick_in.close()
    print(len(data_features))

    print("HOG features saved in {}".format(path_features))



if __name__=='__main__':
    resize_images()
    extract_features()