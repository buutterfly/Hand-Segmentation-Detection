from segmentation_thresholding import *
import warnings
import os

#The skin pixels regions will be extracted from videos that contain only hands 
#as part of the segmentation process that uses OTSU's thresholding method so that we can build the positive dataset.

warnings.filterwarnings('ignore')

video = cv2.VideoCapture(0)
count = 0
path = 'dataset_hands'
for l in range(0, 1):
    images_list = os.listdir(path + "/" + str(l))
    for y in images_list:
        img = cv2.imread(path + "/" + str(l) + "/" + y)
        result, thresh = segmentation_and_thresholding(img)  # Call the segmentation_and_thresholding function
        contours, hierarchy = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )
        for cnt in contours:
            length = cv2.arcLength(cnt, True)
            if length > 50:
                x, y, w, h = cv2.boundingRect(cnt)
                img_tst = img[y:y + h, x:x + w]
                name = './dataset_hands/v/' + str(count) + '.jpg'
                cv2.imwrite(name, img_tst)
                count += 1
