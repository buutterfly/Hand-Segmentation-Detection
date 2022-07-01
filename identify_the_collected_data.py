import os
import cv2
import numpy as np
path = 'dataset_hands'
data = []
label = []

# N.T.: we are doing a binary classification (existance of hands)
for l in range(0, 2):
    images_list = os.listdir(path + "/" + str(l))
    for y in images_list:
        img = cv2.imread(path + "/" + str(l) + "/" + y)
        img = cv2.resize(img, (48, 48))
        data.append(img)
        label.append(l)

data = np.array(data)
label = np.array(label)

np.save('data', data)
np.save('label', label)
