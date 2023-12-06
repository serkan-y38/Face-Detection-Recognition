import os
import cv2 as cv
import numpy as np


def create_train():
    people = ['person1', 'person2']
    features, labels = [], []

    DIR = r'C:\...'
    cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img_label in os.listdir(path):
            img_path = os.path.join(path, img_label)
            img = cv.imread(img_path)

            if img is not None:
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

                for (x, y, w, h) in faces:
                    faces_roi = gray[y:y + h, x:x + w]
                    features.append(faces_roi)
                    labels.append(label)

    return features, labels


create_train = create_train()
features, labels = np.array(create_train[0], dtype='object'), np.array(create_train[1])

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(features, labels)

face_recognizer.save('trained.yml')
