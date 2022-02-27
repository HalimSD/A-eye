# Downloads dogs_vs_cats dataset from Kaggle https://www.kaggle.com/c/dogs-vs-cats/data
# Move each category into its own folder then convert the images to grayscale colors and resize them to 100 * 100
# adding a descriptor file bg.txt for all negative files
import os
import shutil
import cv2


source_dir = './data/train_data/train_data/train'
negatives_path = './data/train_data/negatives'
positives_path = './data/train_data/positives'
cats_target_dir = './data/train_data/negatives/cats'
dogs_target_dir = './data/train_data/negatives/dogs'


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
            cat_image_path = os.path.join(cats_target_dir, cat_image)
            img = cv2.imread(cat_image_path,cv2.IMREAD_GRAYSCALE)
            resized_image = cv2.resize(img, (100, 100))
            cv2.imwrite(cat_image_path,resized_image)
        except Exception as e:
            print(str(e))

def resize_dog_files():
    for dog_image in os.listdir(dogs_target_dir):
        try:
            dog_image_path = os.path.join(dogs_target_dir, dog_image)
            img = cv2.imread(dog_image_path,cv2.IMREAD_GRAYSCALE)
            resized_image = cv2.resize(img, (100, 100))
            cv2.imwrite(dog_image_path,resized_image)
        except Exception as e:
            print(str(e))

def create_pos_n_neg():
    for category in os.listdir(negatives_path):
            if "." not in category:
                for img in os.listdir(os.path.join(negatives_path, category)):
                    try:
                        if "." not in category:
                            line = os.path.join(negatives_path, category)+'/'+img+'\n'
                            with open('bg.txt','a') as f:
                                f.write(line)
                    except Exception as e:
                        print(str(e))

    for category in os.listdir(positives_path):
        if "." not in category:
                for image in os.listdir(os.path.join(positives_path, category)):
                    try:
                        if "." not in category:
                            img = cv2.imread(os.path.join(os.path.join(positives_path, category), image),cv2.IMREAD_GRAYSCALE)
                            resized_image = cv2.resize(img, (20, 20))
                            cv2.imwrite(os.path.join(positives_path,os.path.join(category, image)),resized_image)
                    except Exception as e:
                        print(str(e))

# resize_dog_files()
# resize_cat_files()
create_pos_n_neg()


# COMMAND LINES TO GENERATE POSITIVE IMAGES FOR TRAINING BY PLACING THE POSITIVE IMAGE OVER NEGATIVE IMAGES

# opencv_createsamples -img data/train_data/positives/watch5050.jpg -bg bg.txt -info data/train_data/positives/generated_positives/info.lst -pngoutput data/train_data/positives/generated_positives -maxxangle 0.5 -maxyangle 0.5 -maxzangle 0.5 -num 1950
# opencv_createsamples -info data/train_data/positives/generated_positives/info.lst -num 1950 -w 20 -h 20 -vec positives.vec
# opencv_traincascade -data data/train_data/training_cascade -vec positives.vec -bg bg.txt -numPos 1800 -numNeg 900 -numStages 10 -w 20 -h 20