import glob
import os
import cv2
import joblib
import numpy as np
from PIL import Image
from sklearn.svm import SVC
from sklearn import metrics
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


pos_imgs = 'data/1'
neg_imgs = 'data/0'
pos_feat_path = './data/features/pos'
neg_feat_path = './data/features/neg'


features_list = []
lables_list = []

for file in os.listdir(pos_imgs):
    if ".png" in file:
        try:
            img= Image.open(os.path.join(pos_imgs , file))
            # img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            img = img.convert('L')
            fd = hog(img, orientations = 9, pixels_per_cell=(9, 9), cells_per_block=(2, 2), block_norm='L2', feature_vector=True)
            features_list.append(fd)
            lables_list.append(1)
        except Exception as e:
            print(str(e))

for file in os.listdir(neg_imgs):
    if ".png" in file:
        try:
            img= Image.open(os.path.join(neg_imgs , file))
            img = img.convert('L')
            # img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            fd = hog(img, orientations = 9, pixels_per_cell=(9, 9), cells_per_block=(2, 2), block_norm='L2', feature_vector=True)
            features_list.append(fd)
            lables_list.append(0)
        except Exception as e:
            print(str(e))
# Convert objects to Numpy Objects
# samples = np.float32(features_list)
# labels = np.array(lables_list)

# Shuffle Samples
# rand = np.random.RandomState(321)
# shuffle = rand.permutation(len(samples))
# samples = samples[shuffle]
# labels = labels[shuffle]    


# svm = cv2.ml.SVM_create()
# svm.setType(cv2.ml.SVM_C_SVC)
# svm.setKernel(cv2.ml.SVM_RBF) 
# svm.setGamma(5.383)
# svm.setC(2.67)
# svm.train(samples, cv2.ml.ROW_SAMPLE, labels)
# svm.save('./models/svm_data.dat')


# encode the labels, converting them from strings to integers
le = LabelEncoder()
labels = le.fit_transform(lables_list)


# (trainData, testData, trainLabels, testLabels) = train_test_split(
# 	np.array(features_list), labels, test_size=0.20, random_state=42)

# print(len(testLabels))

# model = LinearSVC()
# model.fit(trainData, trainLabels)
# predictions = model.predict(testData)
# print(classification_report(testLabels, predictions))
# print('Accuracy: ' , metrics.accuracy_score(testLabels, predictions))
# joblib.dump(model, './models/model_name.npy')

kernels = ['Polynomial', 'RBF', 'Sigmoid','Linear']#A function which returns the corresponding SVC model
def getClassifier(ktype):
    if ktype == 0:
        # Polynomial kernal
        return SVC(kernel='poly', degree=8, gamma="auto")
    elif ktype == 1:
        # Radial Basis Function kernal
        return SVC(kernel='rbf', gamma="auto")
    elif ktype == 2:
        # Sigmoid kernal
        return SVC(kernel='sigmoid', gamma="auto")
    elif ktype == 3:
        # Linear kernal
        return SVC(kernel='linear', gamma="auto", max_iter = 14000)

for i in range(4):
    # Separate data into test and training sets
    X_train, X_test, y_train, y_test = train_test_split(np.array(features_list), labels, test_size = 0.20)# Train a SVC model using different kernal
    svclassifier = getClassifier(i) 
    svclassifier.fit(X_train, y_train)# Make prediction
    joblib.dump(svclassifier, './models/{}_model.npy'.format(i))
    y_pred = svclassifier.predict(X_test)# Evaluate our model
    print("Evaluation:", kernels[i], "kernel")
    print(classification_report(y_test,y_pred))