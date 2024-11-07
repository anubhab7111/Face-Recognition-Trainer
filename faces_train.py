import os
import cv2 as cv
import numpy as np

people = []
for i in os.listdir('Resources/Faces/train'):
    people.append(i)

# print(people)

haar_cascade = cv.CascadeClassifier('haar_face.xml')

DIR = "Resources/Faces/train"

features = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            img_array = cv.imread(img_path)
            if img_array is None:
                print(f"Warning: Could not load image {img_path}.")
                continue

            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)

            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()

features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

face_recognizer.train(features, labels)

face_recognizer.save('face_trained.yml')
np.save('featurers.npy', features)
np.save('labels.npy', labels)

print("Training completed and data saved.")