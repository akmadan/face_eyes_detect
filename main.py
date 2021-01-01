import cv2
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('haarcascade_eye.xml')
cap = cv2.VideoCapture('sample.mp4')
while (cap.isOpened()):
    _, frame =cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)
    for x,y,w,h in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 1)
        roi_color = frame[y:y+h, x:x+w]
        roi_gray = frame[y:y+h, x:x+w]
        eyes = eye_classifier.detectMultiScale(roi_gray)
        for ex,ey,ew,eh in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (255,0,0),3)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1)==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()