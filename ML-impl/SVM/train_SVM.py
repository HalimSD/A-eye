import pickle
import random
import joblib
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


pick_in = open('./data/picked.pickle', 'rb')
data = pickle.load(pick_in)
pick_in.close()
random.shuffle(data)

features_list = []
lables_list = []

for feature, label in data:
    features_list.append(feature)
    lables_list.append(label)

kernels = ['Polynomial', 'RBF', 'Sigmoid','Linear']

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
        return SVC(kernel='linear', gamma="auto")

for i in range(4):
    X_train, X_test, y_train, y_test = train_test_split(features_list, lables_list, test_size = 0.20)
    svclassifier = getClassifier(i)
    svclassifier.fit(X_train, y_train)
    joblib.dump(svclassifier, './models/{}_model.npy'.format(i))
    y_pred = svclassifier.predict(X_test)
    print("Evaluation:", kernels[i], "kernel")
    print(classification_report(y_test,y_pred))