import cv2


cap = cv2.VideoCapture("intersection.mp4")

isWorking, frame = cap.read()
x, y, w, h = cv2.selectROI(frame)
print(y, y + h, x, x + w)
