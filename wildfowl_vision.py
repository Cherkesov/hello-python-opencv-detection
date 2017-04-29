# encoding: utf-8
# module wildfowl_vision


def rgb2bgr(rgb):
    return list(reversed(rgb))


def contour_len(cnr):
    l = 0
    for i in range(0, len(cnr) - 1):
        x1, y1 = cnr[i][0]
        x2, y2 = cnr[i + 1][0]
        l += pow(pow(x1 - x2, 2) + pow(y1 - y2, 2), 0.5)
    return l


def filter_long_contours(contours):
    res = list()
    for i in range(0, len(contours)):
        if contour_len(contours[i]) >= 100:
            res.append(contours[i])
    return res
