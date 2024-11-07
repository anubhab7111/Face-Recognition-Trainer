import cv2 as cv

img = cv.imread('Resources/Photos/lady.jpg')
if img is None:
    print("Error: Image not found.")
else:

    # cv.imshow("LadyImage", img)

    # Converting image to GreyScale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # cv.imshow("GreyScale Image", gray)

    #Initializing ourclassifier
    haar_cascade = cv.CascadeClassifier('haar_face.xml')
    if haar_cascade is None:
        print("No classifier found!")

    # Creating rectangles for detected faces
    faces_rect = haar_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3)
    print(f"Faces found: {len(faces_rect)}")

    # Drawing rectangle around faces
for (x, y, w, h) in faces_rect:
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

    cv.imshow('Detected Faces', img)
    cv.waitKey(0)