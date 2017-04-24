import numpy as np
import cv2

# load the games image
image = cv2.imread("images/face-6.jpg")

# upper = np.array([65, 65, 255])
# lower = np.array([0, 0, 200])
# mask = cv2.inRange(image, lower, upper)
# (contours, _) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# (contours, _) = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

if len(contours) == 0:
    print "Expected color not found!"
    exit(0)

print contours
c = max(contours, key=cv2.contourArea)

# peri = cv2.arcLength(c, True)
# approx = cv2.approxPolyDP(c, 0.05 * peri, True)
# cv2.drawContours(image, [approx], -1, (0, 255, 0), 4)

cv2.drawContours(image, contours, -1, (0, 255, 0), 4)
cv2.imshow("Image", image)
cv2.waitKey(0)
