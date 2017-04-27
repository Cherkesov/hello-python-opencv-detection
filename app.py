import numpy as np
import cv2
import itertools


def is_gray(image):
    height, width, channels = image.shape
    for x in range(0, width):
        for y in range(0, height):
            if not image[x, y, 0] == image[x, y, 1] == image[x, y, 2]:
                return False
    return True


def draw_counter_with_colored_mask(image, lower_color, upper_color, counter_color):
    lower = np.array(lower_color)
    upper = np.array(upper_color)
    mask = cv2.inRange(image, lower, upper)
    (contours, _) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return
    cv2.drawContours(image, contours, -1, counter_color, 1)
    pass


def draw_counter_with_gray_threshold(image, threshold_color, counter_color):
    if not is_gray(image):
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = image.copy()
    ret, thresh = cv2.threshold(img_gray, *threshold_color)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return
    cv2.drawContours(image, contours, -1, counter_color, 1)
    pass


def detect_with_haar_cascade(image, haar_cascade):
    return haar_cascade.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    pass


def crop_image_with_haar_cascade(image, haar_cascade):
    dtcs = detect_with_haar_cascade(image, haar_cascade)
    if len(dtcs) == 0:
        return image
    (x, y, w, h) = dtcs[0]
    return image[x:x + w, y:y + h]


def detect_areas_by_signatures_base(image, base_name, extension, names_range):
    # All the 6 methods for comparison in a list
    methods = [
        'cv2.TM_CCOEFF',
        'cv2.TM_CCOEFF_NORMED',
        'cv2.TM_CCORR',
        'cv2.TM_CCORR_NORMED',
        'cv2.TM_SQDIFF',
        'cv2.TM_SQDIFF_NORMED',
    ]

    dtcs = list()

    if not is_gray(image):
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = image.copy()

    # for sig_number in range(0, 3):
    for sig_number in names_range:
        filename = "signature/%s/%d.%s" % (base_name, sig_number, extension)
        print filename
        template = cv2.imread(filename, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        w, h = template.shape[::-1]

        for method_name in methods:
            method = eval(method_name)

            # Apply template Matching
            res = cv2.matchTemplate(img_gray, template, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                top_left = min_loc
            else:
                top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)

            dtcs.append(top_left + bottom_right)

    return dtcs


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

haarcascade_profileface = cv2.CascadeClassifier(cascades_dir + "/haarcascade_profileface.xml")
haarcascade_frontalcatface = cv2.CascadeClassifier(cascades_dir + "/haarcascade_frontalcatface.xml")
haarcascade_frontalface_alt = cv2.CascadeClassifier(cascades_dir + "/haarcascade_frontalface_alt.xml")
haarcascade_frontalface_alt2 = cv2.CascadeClassifier(cascades_dir + "/haarcascade_frontalface_alt2.xml")
haarcascade_frontalface_default = cv2.CascadeClassifier(cascades_dir + "/haarcascade_frontalface_default.xml")
haarcascade_frontalface_alt_tree = cv2.CascadeClassifier(cascades_dir + "/haarcascade_frontalface_alt_tree.xml")
haarcascade_frontalcatface_extended = cv2.CascadeClassifier(cascades_dir + "/haarcascade_frontalcatface_extended.xml")

haarcascade_mcs_mouth = cv2.CascadeClassifier(cascades_dir + "/haarcascade_mcs_mouth.xml")

haarcascade_eye = cv2.CascadeClassifier(cascades_dir + "/haarcascade_eye.xml")
haarcascade_mcs_lefteye = cv2.CascadeClassifier(cascades_dir + "/haarcascade_mcs_lefteye.xml")
haarcascade_mcs_righteye = cv2.CascadeClassifier(cascades_dir + "/haarcascade_mcs_righteye.xml")
haarcascade_lefteye_2splits = cv2.CascadeClassifier(cascades_dir + "/haarcascade_lefteye_2splits.xml")
haarcascade_mcs_eyepair_big = cv2.CascadeClassifier(cascades_dir + "/haarcascade_mcs_eyepair_big.xml")
haarcascade_righteye_2splits = cv2.CascadeClassifier(cascades_dir + "/haarcascade_righteye_2splits.xml")
haarcascade_mcs_eyepair_small = cv2.CascadeClassifier(cascades_dir + "/haarcascade_mcs_eyepair_small.xml")
haarcascade_eye_tree_eyeglasses = cv2.CascadeClassifier(cascades_dir + "/haarcascade_eye_tree_eyeglasses.xml")

# load the games image
im = cv2.imread("images/face-9.jpg")

im = cv2.fastNlMeansDenoisingColored(im, None, 10, 10, 10, 30)
# im = cv2.fastNlMeansDenoising(im, None, 10, 10, 30)
# im = cv2.fastNlMeansDenoisingColored(im, None, 5, 5, 5, 15)


# kernel_sharpen_1 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
# kernel_sharpen_2 = np.array([[1, 1, 1], [1, -7, 1], [1, 1, 1]])
# kernel_sharpen_3 = np.array(
#     [[-1, -1, -1, -1, -1], [-1, 2, 2, 2, -1], [-1, 2, 8, 2, -1], [-1, 2, 2, 2, -1], [-1, -1, -1, -1, -1]]
# ) / 8.0

# applying different kernels to the input image
# im = cv2.filter2D(im, -1, kernel_sharpen_1)
# im = cv2.filter2D(im, -1, kernel_sharpen_2)
# im = cv2.filter2D(im, -1, kernel_sharpen_3)

# im = crop_image_with_haar_cascade(im, haarcascade_frontalface_alt2)
# im = crop_image_with_haar_cascade(im, haarcascade_mcs_mouth)


# haar_face = (
#     haarcascade_profileface,
#     haarcascade_frontalcatface,
#     haarcascade_frontalface_alt,
#     haarcascade_frontalface_alt2,
#     haarcascade_frontalface_default,
#     haarcascade_frontalface_alt_tree,
#     haarcascade_frontalcatface_extended,
# )
# detections = list()
# for hc_face in haar_face:
#     detections.append(detect_with_haar_cascade(im, hc_face))
# for d in detections:
#     (x, y, w, h) = d[0]
#     cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)


eye_haar_cascades = (
    haarcascade_eye,
    haarcascade_mcs_lefteye,
    haarcascade_mcs_righteye,
    haarcascade_lefteye_2splits,
    haarcascade_mcs_eyepair_big,
    haarcascade_righteye_2splits,
    haarcascade_mcs_eyepair_small,
    haarcascade_eye_tree_eyeglasses,
)
detections = list()
for eye_hc in eye_haar_cascades:
    a = detect_with_haar_cascade(im, eye_hc)
    x = [detections, a]
    detections = list(itertools.chain.from_iterable(x))
for d in detections:
    (x, y, w, h) = d
    cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)

# draw_counter_with_gray_threshold(im, (127, 255, 0), [0, 0, 255])  # hair, eyebrows, eyelashes
# draw_counter_with_colored_mask(im, [0, 0, 128], [64, 64, 255], [0, 255, 0])  # mouth input counter
# draw_counter_with_colored_mask(im, [0, 0, 205], [170, 170, 255], [255, 255, 0])  # lips counter
# draw_counter_with_colored_mask(im, [0, 0, 200], [190, 170, 255], [0, 255, 255])  # lips counter


# detections = detect_areas_by_signatures_base(im, "eye-png", "png", range(0, 6))
# for d in detections:
#     cv2.rectangle(im, (d[0], d[1]), (d[2], d[3]), 255, 2)

cv2.imshow("Image", im)
cv2.waitKey(0)
cv2.destroyAllWindows()
