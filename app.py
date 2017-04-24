import numpy as np
import cv2

# load the games image
image = cv2.imread("images/face-9.jpg")

# find the red color game in the image
# upper = np.array([65, 65, 255])
# lower = np.array([0, 0, 200])

# upper = np.array([32, 32, 255])
# lower = np.array([0, 0, 128])

# upper = np.array([200, 120, 110])
# lower = np.array([150, 70, 60])

# upper = np.array([200, 180, 255])
# lower = np.array([150, 70, 60])

# Select dark areas
# upper = np.array([64, 64, 64])
# lower = np.array([0, 0, 0])

#
upper = np.array([240, 120, 120])
lower = np.array([64, 64, 64])

mask = cv2.inRange(image, lower, upper)

# find contours in the masked image and keep the largest one
(cnts, _) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if len(cnts) == 0:
    print "Expected color not found!"
    exit(0)

print cnts
c = max(cnts, key=cv2.contourArea)

# approximate the contour
peri = cv2.arcLength(c, True)
approx = cv2.approxPolyDP(c, 0.05 * peri, True)

# draw a green bounding box surrounding the red game
# cv2.drawContours(image, [approx], -1, (0, 255, 0), 4)
cv2.drawContours(image, cnts, -1, (0, 255, 0), 4)
cv2.imshow("Image", image)
cv2.waitKey(0)
