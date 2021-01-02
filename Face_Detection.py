import cv2
Frontal_Faces= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

video = cv2.VideoCapture("Video.mp4")

while True:
    ret,frame=video.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = Frontal_Faces.detectMultiScale(gray,1.3,3)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)


    cv2.imshow("Faces",frame)
    if cv2.waitKey(1) & 0xff == ord ('q'):
        break

video.release()
cv2.destroyAllWindows()