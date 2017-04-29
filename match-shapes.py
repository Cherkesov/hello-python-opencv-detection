import numpy as np
import cv2
import wildfowl_vision as wv

# get the pictures from the forlder
original_colored = cv2.imread('images/face-11-2.jpg')
drawn_colored = cv2.imread('signature/lips/2.jpg')

original_colored = cv2.fastNlMeansDenoisingColored(original_colored, None, 10, 5, 10, 20)
# drawn_colored = cv2.fastNlMeansDenoisingColored(drawn_colored, None, 10, 10, 10, 30)

# rvO, original_colored = cv2.threshold(original_colored, 127, 255, cv2.THRESH_BINARY)
# rvO, original_colored = cv2.threshold(original_colored, 96, 192, cv2.THRESH_BINARY)
# rvD, drawn = cv2.threshold(drawn, 127, 255, cv2.THRESH_BINARY)

# make them gray
original = cv2.cvtColor(original_colored, cv2.COLOR_BGR2GRAY)
# drawn = cv2.cvtColor(drawn_colored, cv2.COLOR_BGR2GRAY)

# apply erosion
kernel = np.ones((2, 2), np.uint8)
original = cv2.erode(original, kernel, iterations=1)
# drawn = cv2.erode(drawn, kernel, iterations=1)

# retrieve edges with Canny
thresh = 175
# thresh = 10
# original = cv2.Canny(original, thresh, thresh * 2)
original = cv2.Canny(original, 1, 2)
drawn = cv2.Canny(drawn_colored, thresh, thresh * 2)

# original = [[[0, 0, 255 % j] for j in i] for i in original]
# dt = np.dtype('f8')
# original = np.array(original, dtype=dt)

# cv2.imshow("Image 122", original)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# exit(0)

# extract contours
# originalContours, Orighierarchy = cv2.findContours(original, cv2.cv.CV_RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# drawnContours, Drawnhierarchy = cv2.findContours(drawn, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

originalContours, Orighierarchy = cv2.findContours(original, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
drawnContours, Drawnhierarchy = cv2.findContours(drawn, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

originalContours = wv.filter_long_contours(originalContours)

# colors = (
#     (0, 255, 255),
#     (0, 255, 0),
#     (255, 0, 255),
#     (255, 255, 0),
# )
# for i in range(0, len(drawnContours)):
#     cv2.drawContours(drawn_colored, [drawnContours[i]], -1, colors[i], 2)
# cv2.imshow("Image 122", drawn_colored)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# exit(0)

founded_counters = list()
for dr_i in range(0, len(drawnContours)):
    for or_i in range(0, len(originalContours)):
        match_result = cv2.matchShapes(
            drawnContours[dr_i],
            originalContours[or_i],
            cv2.cv.CV_CONTOURS_MATCH_I1,
            0.0
        )
        # print match_result
        if match_result == 0:
            cntr = originalContours[or_i]
            # hull = cv2.convexHull(cntr, returnPoints=False)
            # if len(cntr) > 3 and len(hull):
            #     defects = cv2.convexityDefects(cntr, hull)
            #     for i in range(defects.shape[0]):
            #         s, e, f, d = defects[i, 0]
            #         start = tuple(cntr[s][0])
            #         end = tuple(cntr[e][0])
            #         far = tuple(cntr[f][0])
            #         cv2.line(original_colored, start, end, [0, 255, 0], 2)
            #         # cv2.circle(original_colored, far, 5, [0, 0, 255], -1)
            #
            #         # print (s, e, f, d)
            #         # print cntr[f]

            founded_counters.append(cntr)

print len(founded_counters)

cv2.drawContours(original_colored, originalContours, -1, (255, 255, 255), 1)
cv2.drawContours(original_colored, founded_counters, -1, (0, 255, 0), 2)
cv2.imshow("Image 123", original_colored)

# cv2.imshow("Image 234", original)

# cv2.drawContours(drawn_colored, drawnContours, -1, (0, 255, 255), 2)
# cv2.imshow("Image 122", drawn_colored)

cv2.waitKey(0)
cv2.destroyAllWindows()
