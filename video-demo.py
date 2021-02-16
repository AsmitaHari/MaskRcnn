import numpy as np
import cv2 as cv
from visualize_cv2 import model, display_instances, class_names

# print(cv.__version__)
# cap = cv.VideoCapture('Friends.mkv')
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    results = model.detect([frame], verbose=1)

    # Visualize results
    r = results[0]
    masked_image = display_instances(frame, r['rois'], r['masks'], r['class_ids'],
                                     class_names, r['scores'])
    cv.imshow("masked_image", masked_image)
    if (cv.waitKey(1) & 0xFF == ord('q')):
        break
cap.release()
cv.destroyAllWindows()