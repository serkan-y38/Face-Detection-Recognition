import cv2 as cv

cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
people = ['person1', 'person2']

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('trained.yml')


def recognize(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        faces_roi = gray[y:y + h, x:x + w]

        label, confidence = face_recognizer.predict(faces_roi)
        # print(confidence)

        cv.putText(img, str(people[label]), (20, 20), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), thickness=1)
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), thickness=2)

    cv.imshow('recognized face', img)
    cv.waitKey(0)
    cv.destroyAllWindows()


test_img = cv.imread('test.jpg')
recognize(test_img)
