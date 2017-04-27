import numpy as np
import cv2

# get the pictures from the forlder
original = cv2.imread('images/face-1.jpg')
drawn = cv2.imread('signature/lips/0.jpg')

# make them gray
originalGray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
drawnGray = cv2.cvtColor(drawn, cv2.COLOR_BGR2GRAY)

# apply erosion
kernel = np.ones((2, 2), np.uint8)
originalErosion = cv2.erode(originalGray, kernel, iterations=1)
drawnErosion = cv2.erode(drawnGray, kernel, iterations=1)

# retrieve edges with Canny
thresh = 175
originalEdges = cv2.Canny(originalErosion, thresh, thresh * 2)
drawnEdges = cv2.Canny(drawnErosion, thresh, thresh * 2)

# extract contours
originalContours, Orighierarchy = cv2.findContours(originalEdges, cv2.cv.CV_RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
drawnContours, Drawnhierarchy = cv2.findContours(drawnEdges, cv2.cv.CV_RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(drawnEdges, drawnContours, 0, (255, 0, 0), 3)
cv2.imshow("Image 2", drawnEdges)
cv2.waitKey(0)
exit(0)

# print cv2.matchShapes(drawnContours[0], originalContours[0], cv2.cv.CV_CONTOURS_MATCH_I1, 0.0)

founded_counters = list()
for dr_i in range(0, len(drawnContours)):
    for or_i in range(0, len(originalContours)):
        print or_i, dr_i
        # print originalContours[or_i]
        # print drawnContours[dr_i]
        match_result = cv2.matchShapes(drawnContours[dr_i], originalContours[or_i], cv2.cv.CV_CONTOURS_MATCH_I1, 0.0)
        print match_result
        print "=================================================="

        if match_result < 0.3:
            founded_counters.append(originalContours[or_i])

cv2.drawContours(original, founded_counters, 0, (0, 255, 0), 3)
cv2.imshow("Image", original)

cv2.waitKey(0)
cv2.destroyAllWindows()