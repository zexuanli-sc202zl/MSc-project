import os, cv2, math, random
import numpy as np
from PIL import Image

# for idx, i in enumerate(os.listdir('data/training/images')):
#     os.rename('data/training/images/{}'.format(i), 'data/training/images/{}.tif'.format(idx))
#     os.rename('data/training/1st_manual/{}_manual1.gif'.format(i.split('_')[0]), 'data/training/1st_manual/{}.gif'.format(idx))

def random_crop(img, mask, scale=[0.8, 1.0], ratio=[3. / 4., 4. / 3.]):
    aspect_ratio = math.sqrt(np.random.uniform(*ratio))
    w = 1. * aspect_ratio
    h = 1. / aspect_ratio
    src_h, src_w = img.shape[:2]

    bound = min((float(src_w) / src_h) / (w ** 2),
                (float(src_h) / src_w) / (h ** 2))
    scale_max = min(scale[1], bound)
    scale_min = min(scale[0], bound)

    target_area = src_h * src_w * np.random.uniform(scale_min, scale_max)
    target_size = math.sqrt(target_area)
    w = int(target_size * w)
    h = int(target_size * h)

    i = np.random.randint(0, src_w - w + 1)
    j = np.random.randint(0, src_h - h + 1)

    img = img[j:j + h, i:i + w]
    mask = mask[j:j + h, i:i + w]
    return img, mask

def flip(img, mask):
    """
    翻转
    :param img:
    :param mode: 1=水平翻转 / 0=垂直 / -1=水平垂直
    :return:
    """
    mode = np.random.choice([-1, 0, 1])
    return cv2.flip(img, flipCode=mode), cv2.flip(mask, flipCode=mode)

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

def adjust_contrast_bright(img, contrast=1.2, brightness=100):
    """
    调整亮度与对比度
    dst = img * contrast + brightness
    :param img:
    :param contrast: 对比度   越大越亮
    :param brightness: 亮度  0~100
    :return:
    """
    # 像素值会超过0-255， 因此需要截断
    return np.uint8(np.clip((contrast * img + brightness), 0, 255))

for i in range(20, 100):
    img_id = np.random.randint(0, 20)

    img = cv2.imread('data/training/images/{}.tif'.format(img_id))
    mask = np.array(Image.open('data/training/1st_manual/{}.gif'.format(img_id)))

    random_id = np.random.randint(1, 5)

    if random_id == 1:
        img, mask = random_crop(img, mask)
    elif random_id == 2:
        img, mask = flip(img, mask)
    elif random_id == 3:
        img = random_noise(img)
    elif random_id == 4:
        img = adjust_contrast_bright(img)

    mask = Image.fromarray(mask, mode='L')
    cv2.imwrite('data/training/images/{}.tif'.format(i), img)
    mask.save('data/training/1st_manual/{}.gif'.format(i))