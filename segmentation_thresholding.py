import cv2
import numpy as np
import warnings

warnings.filterwarnings('ignore')

#In this file you will find removal of facial regions and segmentation function.
#A manual implementation of OTSU's thresholding algorithm was used to serve in the segmentation process.

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def fillHoles(mask):
    removal_face = mask.copy()
    h, w = removal_face.shape[:2]
    maskTemp = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(removal_face, maskTemp, (0, 0), 255)
    mask2 = cv2.bitwise_not(removal_face)

    return mask2 | mask


def threshold_ots_algorithm(bl):
    hist = cv2.calcHist([bl], [0], None, [256], [0, 256])  # Calculate histogram
    hist_norm = hist.ravel() / hist.max()
    Q = hist_norm.cumsum()
    bins = np.arange(256)  # Array / Total number of bins in histogram
    fn_min = np.inf
    thresh = -1
    for i in range(1, 256):
        p1, p2 = np.hsplit(hist_norm, [i])  # Probabilities
        q1, q2 = Q[i], Q[255] - Q[i]  # Cum Cum of classes
        w1, w2 = np.hsplit(bins, [i])  # Weights

        # Means & Variances of Background & Foreground
        m1, m2 = np.sum(p1 * w1) / q1, np.sum(p2 * w2) / q2
        v1, v2 = np.sum(((w1 - m1) ** 2) * p1) / q1, np.sum(((w2 - m2) ** 2) * p2) / q2

        # calculates the minimization function
        fn = v1 * q1 + v2 * q2
        if fn < fn_min:
            fn_min = fn
            thresh = i
    return thresh


def generated_image(img, threshold):
    H, W = img.shape
    X = np.zeros((H, W), dtype="uint8")
    for i in range(0, H):
        for j in range(0, W):
            if img[i, j] >= threshold:
                X[i, j] = 255
            else:
                X[i, j] = 0

    return X


def face_removal_function(img):
    img = cv2.cvtColor(img, cv2.IMREAD_COLOR)
    imgOut = img.copy()
    faces = faceCascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(100, 100))
    for (x, y, w, h) in faces:
        # Extract face from the image
        face = img[y:y + h, x:x + w]
        hsi = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)  # HSI COLOR MODEL
        lower = np.array([0, 40, 0], dtype="uint8")
        upper = np.array([25, 255, 255], dtype="uint8")
        mask = cv2.inRange(hsi, lower, upper)
        A = face[:, :, 1]
        B = face[:, :, 2]
        AB = cv2.add(A, B)
        mask = fillHoles(mask)
        mask = cv2.dilate(mask, None, anchor=(-1, -1), iterations=3, borderType=1, borderValue=1)
        mean = AB // 2
        mask = mask.astype(np.bool)[:, :, np.newaxis]
        mean = mean[:, :, np.newaxis]
        FaceOut = face.copy()
        np.copyto(FaceOut, mean, where=mask)
        imgOut[y:y + h, x:x + w, :] = FaceOut
    return imgOut


def segmentation_and_thresholding(img):
    img = face_removal_function(img)
    hsi = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # HSI COLOR MODEL
    ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)  # YCBCR COLOR MODEL

    # SKIN color range for both HSI and YCBCR,it might need to change based on race etc.
    hlower = np.array([0, 40, 0], dtype="uint8")
    hupper = np.array([25, 255, 255], dtype="uint8")
    ylower = np.array((0, 138, 67), dtype="uint8")
    yupper = np.array((255, 173, 133), dtype="uint8")

    # Binary MASK (extracting the skin color image parts)
    ymask = cv2.inRange(ycbcr, ylower, yupper)
    hmask = cv2.inRange(hsi, hlower, hupper)

    binary_mask_image = cv2.bitwise_and(hmask, ymask)  # Bitwise the hmask and ymask.
    blur_median_img = cv2.medianBlur(binary_mask_image, 9)  # Median Blur

    # Morphological operations
    erode_masked_img = cv2.erode(blur_median_img, None, iterations=2)  # Remove noise
    dilate_img = cv2.dilate(erode_masked_img, None, iterations=1)  # dilate the image to increase the hand area

    blur2_masked_image = cv2.blur(dilate_img, (10, 10))  # Blur the image
    thr_value = threshold_ots_algorithm(blur2_masked_image)  # Get the value by otsu's thresholding algorithm.
    thr_img = generated_image(blur2_masked_image, thr_value)  # Get the generated image by the otsu's thresholding.

    result = cv2.bitwise_and(img, img, mask=thr_img)  # Bitwise the generated mask with the image.
    return result, thr_img
