import cv2
from skimage.feature import hog
import joblib
import os

clf = joblib.load(os.path.join('./models/3_model.npy'))
orientations = 9
threshold = .90
pixels_per_cell = [6,6]
cells_per_block = [2, 2]

cap = cv2.VideoCapture(0)
while 1:
    ret, frame = cap.read()
    cv2.imshow('Webcam', frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64,128), interpolation = cv2.INTER_AREA)
    features =  hog(gray, orientations=orientations, pixels_per_cell=pixels_per_cell,cells_per_block=cells_per_block,block_norm='L2', transform_sqrt=False,feature_vector=True)
    features = features.reshape(1, -1)
    pred = clf.predict(features)

    if pred == 1:
        if clf.decision_function(features) > threshold:
            print("Human detected")

    if cv2.waitKey(30)  == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()