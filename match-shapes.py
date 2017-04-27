import numpy as np
import cv2

# get the pictures from the forlder
originalColored = cv2.imread('images/face-1.jpg')
drawnColored = cv2.imread('signature/lips/0.jpg')

originalColored = cv2.fastNlMeansDenoisingColored(originalColored, None, 10, 10, 10, 30)
drawnColored = cv2.fastNlMeansDenoisingColored(drawnColored, None, 10, 10, 10, 30)

# make them gray
original = cv2.cvtColor(originalColored, cv2.COLOR_BGR2GRAY)
drawn = cv2.cvtColor(drawnColored, cv2.COLOR_BGR2GRAY)

# apply erosion
kernel = np.ones((2, 2), np.uint8)
original = cv2.erode(original, kernel, iterations=1)
drawn = cv2.erode(drawn, kernel, iterations=1)

# retrieve edges with Canny
thresh = 175
# thresh = 10
# original = cv2.Canny(original, thresh, thresh * 2)
original = cv2.Canny(original, 1, 1)
drawn = cv2.Canny(drawn, thresh, thresh * 2)

# extract contours
# originalContours, Orighierarchy = cv2.findContours(original, cv2.cv.CV_RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
originalContours, Orighierarchy = cv2.findContours(original, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# drawnContours, Drawnhierarchy = cv2.findContours(drawn, cv2.cv.CV_RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
drawnContours, Drawnhierarchy = cv2.findContours(drawn, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# cv2.drawContours(original, originalContours, 0, (255, 255, 255), 3)
# cv2.imshow("Image 2", original)
# cv2.waitKey(0)
# exit(0)


# cv2.drawContours(drawn, drawnContours, 0, (255, 0, 0), 3)
# cv2.drawContours(drawn, [drawnContours[0]], 0, (255, 0, 0), 3)
# cv2.drawContours(drawn, [drawnContours[1]], 0, (255, 255, 255), 3)
# cv2.imshow("Image 2", drawn)
# cv2.waitKey(0)
# exit(0)

# print cv2.matchShapes(drawnContours[0], originalContours[0], cv2.cv.CV_CONTOURS_MATCH_I1, 0.0)

print len(originalContours)
print len(drawnContours)

founded_counters = list()
for or_i in range(0, len(originalContours)):
    for dr_i in range(1, len(drawnContours)):
        match_result = cv2.matchShapes(drawnContours[dr_i], originalContours[or_i], cv2.cv.CV_CONTOURS_MATCH_I1, 0.0)
        print match_result
        # if match_result > 300:
        if match_result == 0:
            founded_counters.append(originalContours[or_i])

print len(founded_counters)

cv2.drawContours(originalColored, originalContours, -1, (0, 255, 255), 4)
cv2.drawContours(originalColored, founded_counters, -1, (0, 255, 0), 2)
cv2.imshow("Image 123", originalColored)

# cv2.imshow("Image 234", original)

cv2.drawContours(drawnColored, [drawnContours[1]], -1, (0, 255, 255), 2)
cv2.imshow("Image 122", drawnColored)

cv2.waitKey(0)
cv2.destroyAllWindows()
