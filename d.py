import torch, cv2, os, tqdm, shutil
import matplotlib.pylab as plt
from PIL import Image
import numpy as np
import torchvision

from model import UNet

if __name__ == '__main__':
    model_UNet = torch.load('RepUNet.pkl')
    for module in model_UNet.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()

    new_model = torchvision.models._utils.IntermediateLayerGetter(model_UNet, {'first_block':'first_block',
                                                                               'downblock1':'downblock1',
                                                                               'downblock2':'downblock2',
                                                                               'downblock3':'downblock3',
                                                                               'downblock4':'downblock4'})

    img_show = cv2.imread('data/test/images/1.tif')

    img_shape = img_show.shape
    img = cv2.resize(img_show, (256, 256))
    img = cv2.split(img)[1]
    img = np.array(img, dtype=np.float) / 255.0

    img = np.expand_dims(img, axis=-1)
    img_unet = np.transpose(np.expand_dims(img, axis=0), (0, 3, 1, 2))
    img_unet = torch.from_numpy(img_unet).float()
    pred = new_model(img_unet)
    print([(k, v.shape) for k, v in pred.items()])

    plt.figure(figsize=(10, 10))
    first_block = pred['first_block'][0].detach().numpy()
    first_block_min, first_block_max = np.min(first_block), np.max(first_block)
    first_block = (first_block - first_block_min) / (first_block_max - first_block_min)
    for i in range(len(first_block)):
        plt.subplot(8, 8, 1 + i)
        plt.imshow(first_block[i], cmap=plt.cm.gray)
        plt.axis('off')
    plt.show()

    plt.figure(figsize=(10, 10))
    downblock1 = pred['downblock1'][0].detach().numpy()
    downblock1_min, downblock1_max = np.min(downblock1), np.max(downblock1)
    downblock1 = (downblock1 - downblock1_min) / (downblock1_max - downblock1_min)
    for i in range(64):
        plt.subplot(8, 8, 1 + i)
        plt.imshow(downblock1[i], cmap=plt.cm.gray)
        plt.axis('off')
    plt.show()

    plt.figure(figsize=(10, 10))
    downblock2 = pred['downblock2'][0].detach().numpy()
    downblock2_min, downblock2_max = np.min(downblock2), np.max(downblock2)
    downblock2 = (downblock2 - downblock2_min) / (downblock2_max - downblock2_min)
    for i in range(64):
        plt.subplot(8, 8, 1 + i)
        plt.imshow(downblock2[i], cmap=plt.cm.gray)
        plt.axis('off')
    plt.show()

    plt.figure(figsize=(10, 10))
    downblock3 = pred['downblock3'][0].detach().numpy()
    downblock3_min, downblock3_max = np.min(downblock3), np.max(downblock3)
    downblock3 = (downblock3 - downblock3_min) / (downblock3_max - downblock3_min)
    for i in range(64):
        plt.subplot(8, 8, 1 + i)
        plt.imshow(downblock3[i], cmap=plt.cm.gray)
        plt.axis('off')
    plt.show()

    plt.figure(figsize=(10, 10))
    downblock4 = pred['downblock4'][0].detach().numpy()
    downblock4_min, downblock4_max = np.min(downblock4), np.max(downblock4)
    downblock4 = (downblock4 - downblock4_min) / (downblock4_max - downblock4_min)
    for i in range(64):
        plt.subplot(8, 8, 1 + i)
        plt.imshow(downblock4[i], cmap=plt.cm.gray)
        plt.axis('off')
    plt.show()