import cv2, random
import numpy as np
from PIL import Image
import matplotlib.pylab as plt

def random_noise(img, rand_range=(3, 20)):
    """
    随机噪声
    :param img:
    :param rand_range: (min, max)
    :return:
    """
    img = np.asarray(img, np.float)
    sigma = random.randint(*rand_range)
    nosie = np.random.normal(0, sigma, size=img.shape)
    img += nosie
    img = np.uint8(np.clip(img, 0, 255))
    return img

# img = cv2.imread('data/train/images/1.tif')
# img = random_noise(img)
# mask = np.array(Image.open('data/train/1st_manual/1.gif'))
#
# plt.figure(figsize=(10, 5))
#
# plt.subplot(1, 2, 1)
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.axis('off')
#
# plt.subplot(1, 2, 2)
# plt.imshow(mask, cmap=plt.cm.gray)
# plt.axis('off')
#
# plt.show()

img = cv2.imread('data/train/images/14.tif')

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(cv2.split(img)[0], cmap=plt.cm.gray)
plt.axis('off')
plt.title('B')

plt.subplot(1, 3, 2)
plt.imshow(cv2.split(img)[1], cmap=plt.cm.gray)
plt.axis('off')
plt.title('G')

plt.subplot(1, 3, 3)
plt.imshow(cv2.split(img)[2], cmap=plt.cm.gray)
plt.axis('off')
plt.title('R')

plt.show()