import cv2 as cv


def photo_detect(img):
    cv.imshow("img", img)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h,) in faces:
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)

    cv.imshow("detected", img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def camera_detect():
    camera = cv.VideoCapture(0)
    cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

    while True:
        ret, capture = camera.read()

        gray = cv.cvtColor(capture, cv.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h,) in faces:
            cv.rectangle(capture, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv.imshow("camera", capture)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()


# camera_detect()

img = cv.imread('people.jpg')
photo_detect(img)
