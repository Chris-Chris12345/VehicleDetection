import cv2
import time

Haarcascade = "C:\OpenCV with python\VeDetection\cars.xml"
car_cascade = cv2.CascadeClassifier(Haarcascade)

if car_cascade.empty():
    raise Exception("Error in loading the file.")

vid = "C:\OpenCV with python\VeDetection\car-video.mp4"
cap = cv2.VideoCapture(vid)

if not cap.isOpened():
    raise Exception("The video is not loading properly.")

#Variable for detection logic
Car_detected = False
Detection_time = None

while cap.isOpened():
    ret,frames = cap.read() #ret is True if frame is read sucessfully, frame is the actual image frame
    if not ret:
        break

    gray = cv2.cvtColor(frames,cv2.COLOR_BGR2GRAY) #Any kind of detection works better on a grayscale image

    #Main code to detect the car
    cars = car_cascade.detectMultiScale(gray,scaleFactor = 1.1, minNeighbors = 3) #gray is the input image/video, scaleFactor helps detect the car at different distances(1.1 will reduce the image size by 10% at each scale, minNeighbors (higher value = fewer false detection, lower value = more detection but more errors))
    if len(cars) > 0 and not Car_detected:
        Car_detected = True
        Detection_time = time.time()
        print("Car detected")

    #Draw rectangles around detected cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frames, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frames, "Car detected", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Car Detection", frames)

    if Car_detected and time.time() - Detection_time > 5:
        break
    
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
