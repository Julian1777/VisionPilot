import cv2
import numpy as np

img_path = r"C:\Users\user\Documents\github\self-driving-car-simulation\images\carla\miami_src.png"
img = cv2.imread(img_path)
if img is None:
    print("Image not found!")
    exit(1)

points = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Select src points", img)
        if len(points) == 4:
            print("Selected src points (for perspective transform):")
            print(np.array(points, dtype=np.float32))
            cv2.destroyAllWindows()

cv2.imshow("Select src points", img)
cv2.setMouseCallback("Select src points", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()