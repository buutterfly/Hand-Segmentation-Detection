from segmentation_thresholding import *
from keras.models import load_model
import warnings

warnings.filterwarnings('ignore')

font = cv2.FONT_HERSHEY_COMPLEX

model = load_model('HandTrainingModel.h5')


def preprocessing(img_tr):
    img_tr = cv2.resize(img_tr, (48, 48))
    img_tr = cv2.cvtColor(img_tr, cv2.COLOR_BGR2GRAY)
    normalized = img_tr / 255.0
    reshaped = np.reshape(normalized, (1, 48, 48, 1))
    return reshaped


video = cv2.VideoCapture(0)

while True:
    ret, img = video.read()
    if img is None:
        break
    result, thresh = segmentation_and_thresholding(img)  # Call the segmentation_and_thresholding function
    contours, hierarchy = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )
    for cnt in contours:
        l = cv2.arcLength(cnt, True)
        if l > 50:
            x, y, w, h = cv2.boundingRect(cnt)
            img_tst = img[y:y + h, x:x + w]
            img_tst = preprocessing(img_tst)
            prediction = model.predict(img_tst) #Feed the preprocessed input image to the trained CNN model to confirm whether the detected and segmented pixels are from a hand or not.
            classIndex = np.argmax(prediction, axis=1)
            probability = (np.amax(prediction) * 100)
            if probability >= 80:
                if classIndex == 0:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)
                    cv2.rectangle(img, (x, y - 40), (x + w, y), (0, 0, 0), -2)
                    cv2.putText(img, "Hand Detected!" + str("{:.2f}".format(probability)) + "%", (x, y - 10), font, 0.8,
                                (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Segmentation Result", thresh)
        cv2.imshow("Original Video", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
