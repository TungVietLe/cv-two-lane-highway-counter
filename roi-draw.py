import cv2


cap = cv2.VideoCapture("tophighway.mp4")

isWorking, frame = cap.read()
x, y, w, h = cv2.selectROI(frame)
print(f"{y}, {y + h}, {x}, {x + w}")
