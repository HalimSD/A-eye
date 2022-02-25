# Downloads dogs_vs_cats dataset from Kaggle https://www.kaggle.com/c/dogs-vs-cats/data
# Move each category into its own folder then convert them to grayscale colors and resize them to 100 * 100

import os
import shutil
import cv2


source_dir = './data/train_data/train'
cats_target_dir = './data/train_data/cats'
dogs_target_dir = './data/train_data/dogs'

files = os.listdir(source_dir)
# cat_files = [os.remove(os.path.join(os.getcwd(), cat_file)) 
# for cat_file in os.listdir(os.getcwd()):
#      if 'cat' in cat_file:
#         os.remove(os.path.join(os.getcwd(), cat_file))
# print(os.getcwd())

dog_files = [shutil.move(os.path.join(source_dir, dog_file), dogs_target_dir) for dog_file in files if 'dog' in dog_file]
cat_files = [shutil.move(os.path.join(source_dir, cat_file), cats_target_dir) for cat_file in files if 'cat' in cat_file]

def resize_cat_files():
    for cat_image in os.listdir(cats_target_dir):
        try:
            cat_image_path = os.path.join('./data/train_data/cats', cat_image)
            img = cv2.imread(cat_image_path,cv2.IMREAD_GRAYSCALE)
            resized_image = cv2.resize(img, (100, 100))
            cv2.imwrite(cat_image_path,resized_image)
        except Exception as e:
            print(str(e))

def resize_dog_files():
    for dog_image in os.listdir(dogs_target_dir):
        try:
            dog_image_path = os.path.join('./data/train_data/dogs', dog_image)
            img = cv2.imread(dog_image_path,cv2.IMREAD_GRAYSCALE)
            resized_image = cv2.resize(img, (100, 100))
            cv2.imwrite(dog_image_path,resized_image)
        except Exception as e:
            print(str(e))

resize_dog_files()
resize_cat_files()