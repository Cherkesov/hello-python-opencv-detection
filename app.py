import numpy as np
import cv2


# B G R

def draw_counter_with_colored_mask(image, lower_color, upper_color, counter_color):
    lower = np.array(lower_color)
    upper = np.array(upper_color)
    mask = cv2.inRange(image, lower, upper)
    (contours, _) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return
    cv2.drawContours(image, contours, -1, counter_color, 1)
    # cv2.drawContours(image, contours, -1, (0, 0, 255), 1)
    pass


def draw_counter_with_gray_threshold(image, threshold_color, counter_color):
    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, *threshold_color)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return
    cv2.drawContours(image, contours, -1, counter_color, 1)
    pass


# European woman skin color palette
# (189, 224, 255)
# (148, 205, 255)
# (134, 192, 234)
# (96, 173, 255)
# (159, 227, 255)

# European woman lips color palette
# (238, 193, 173)
# (219, 172, 152)
# (210, 153, 133)
# (201, 130, 118)
# (227, 93, 106)


# 270 240 230

# 440 300
# 520 340

cascades_dir = "/usr/local/share/OpenCV/haarcascades"
haarcascade_frontalface_default = cv2.CascadeClassifier(cascades_dir + "/haarcascade_frontalface_default.xml")
haarcascade_frontalface_alt = cv2.CascadeClassifier(cascades_dir + "/haarcascade_frontalface_alt.xml")
haarcascade_frontalface_alt2 = cv2.CascadeClassifier(cascades_dir + "/haarcascade_frontalface_alt2.xml")
cascade_mouth = cv2.CascadeClassifier(cascades_dir + "/haarcascade_mcs_mouth.xml")

# load the games image
im = cv2.imread("images/face-9.jpg")
# im = cv2.imread("tmp-output/test-2.jpg")

im = cv2.fastNlMeansDenoisingColored(im, None, 10, 10, 10, 30)
# im = cv2.fastNlMeansDenoisingColored(im, None, 5, 5, 5, 15)


# generating the kernels
# kernel_sharpen_1 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
kernel_sharpen_1 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
kernel_sharpen_2 = np.array([[1, 1, 1], [1, -7, 1], [1, 1, 1]])
kernel_sharpen_3 = np.array(
    [[-1, -1, -1, -1, -1], [-1, 2, 2, 2, -1], [-1, 2, 8, 2, -1], [-1, 2, 2, 2, -1], [-1, -1, -1, -1, -1]]
) / 8.0

# applying different kernels to the input image
im = cv2.filter2D(im, -1, kernel_sharpen_1)
# im = cv2.filter2D(im, -1, kernel_sharpen_2)
# im = cv2.filter2D(im, -1, kernel_sharpen_3)

# faces = haarcascade_frontalface_alt2.detectMultiScale(
#     im,
#     scaleFactor=1.1,
#     minNeighbors=5,
#     minSize=(30, 30),
#     flags=cv2.CASCADE_SCALE_IMAGE
# )
# if len(faces) == 0:
#     print "Faces not found!"
#     exit(0)
# (x, y, w, h) = faces[0]
# im = im[x:x + w, y:y + h]


# mouths = cascade_mouth.detectMultiScale(
#     im,
#     scaleFactor=1.1,
#     minNeighbors=5,
#     minSize=(30, 30),
#     flags=cv2.CASCADE_SCALE_IMAGE
# )
# if len(mouths) == 0:
#     print "Mouths not found!"
#     exit(0)
# (x, y, w, h) = mouths[0]
# im = im[x:x + w, y:y + h]


# draw_counter_with_gray_threshold(im, (127, 255, 0), [0, 0, 255])  # hair, eyebrows, eyelashes
# draw_counter_with_colored_mask(im, [0, 0, 128], [64, 64, 255], [0, 255, 0])  # mouth input counter
# draw_counter_with_colored_mask(im, [0, 0, 205], [170, 170, 255], [255, 255, 0])  # lips counter
# draw_counter_with_colored_mask(im, [0, 0, 200], [190, 170, 255], [0, 255, 255])  # lips counter

cv2.imshow("Image", im)
cv2.waitKey(0)
