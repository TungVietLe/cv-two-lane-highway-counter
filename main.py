import cv2
from tracker import *
import numpy as np


class Config:
    def __init__(self) -> None:
        self.history = 100  # how long a thing has to stand still to skip detection
        self.varThreshold = 30
        self.minArea = 20  # only area bigger than this value get new id

        # mask color region that get accepted
        self.maskMin = 254
        self.maskMax = 255


# Create tracker object
tracker = EuclideanDistTracker()
# Create config object
config = Config()

cap = cv2.VideoCapture("intersection.mp4")


# Object detection from Stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(
    history=config.history, varThreshold=config.varThreshold
)


while True:
    isWorking, frame = cap.read()
    height, width, channel = frame.shape

    # Extract Region of interest
    draw = [284, 370, 594, 775]
    roi = frame[draw[0] : draw[1], draw[2] : draw[3]]

    # 1. Object Detection
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, config.maskMin, config.maskMax, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        # Calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area > config.minArea:
            # cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)

            detections.append([x, y, w, h])

    # 2. Object Tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(
            frame,
            str(id),
            (x + draw[2], y + draw[0]),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            (255, 0, 0),
            2,
        )
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("roi", roi)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    if cv2.waitKey(0) == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
