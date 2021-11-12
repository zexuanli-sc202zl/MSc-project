import torch, cv2, os, tqdm, shutil
import matplotlib.pylab as plt
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix

from model import UNet

def get_data(mode):
    path, x, y = [], [], []
    for i in tqdm.tqdm(os.listdir('data/{}/images'.format(mode))):
        img = cv2.imread('data/{}/images/{}'.format(mode, i))
        mask = np.array(Image.open('data/{}/1st_manual/{}.gif'.format(mode, i.split('.')[0])))

        mask[mask != 0] = 1

        x.append(img)
        y.append(mask)
        path.append(i)

    x, y = np.array(x), np.array(y)

    print('success load {} set'.format(mode))
    return path, x, y

def metrice(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    pa = np.diag(cm).sum() / (cm.sum() + 1e-7)

    mpa_arr = np.diag(cm) / (cm.sum(axis=1) + 1e-7)
    mpa = np.nanmean(mpa_arr)

    MIoU = np.diag(cm) / (np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm) + 1e-7)
    MIoU = np.nanmean(MIoU)

    return pa, mpa, MIoU

if __name__ == '__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(DEVICE)

    model_UNet = torch.load('UNet.pkl')
    model_UNet.to(DEVICE)

    path, x, y = get_data('test')

    if os.path.exists('unet-result'):
        shutil.rmtree('unet-result')
    os.mkdir('unet-result')

    for i in range(len(path)):
        img_show = x[i]
        mask = y[i]

        img_shape = img_show.shape
        img = cv2.resize(img_show, (256, 256))
        img = np.array(img, dtype=np.float) / 255.0

        img_unet = np.transpose(np.expand_dims(img, axis=0), (0, 3, 1, 2))
        img_unet = torch.from_numpy(img_unet).to(DEVICE).float()
        pred_unet = np.argmax(model_UNet(img_unet).cpu().detach().numpy()[0], axis=0).reshape((-1))

        mask_true = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST).reshape((-1))

        unet_pa, unet_mpa, unet_miou = metrice(mask_true, pred_unet)

        pred_unet = np.array(pred_unet, dtype=np.uint8).reshape((256, 256))
        pred_unet_img = cv2.resize(pred_unet, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_NEAREST)

        pred_unet_img[pred_unet_img == 1] = 255
        mask[mask == 1] = 255

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(mask, cmap=plt.cm.gray)
        plt.title('true')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(pred_unet_img, cmap=plt.cm.gray)
        plt.title('unet-pred\npa:{:.3f} mpa:{:.3f} miou:{:.3f}'.format(unet_pa, unet_mpa, unet_miou))
        plt.axis('off')

        plt.savefig('unet-result/{}'.format(i))