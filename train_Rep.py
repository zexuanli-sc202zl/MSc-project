import torch, time, datetime, tqdm, os, cv2
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image
import numpy as np

from model import UNet, RepUNet
from sklearn.metrics import confusion_matrix

from typing import Optional
from functools import partial

import torch
from torch.nn.modules.loss import _Loss

def focal_loss_with_logits(
    output: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    alpha: Optional[float] = 0.25,
    normalized: bool = False,
    reduced_threshold: Optional[float] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    target = target.type(output.type())

    logpt = F.binary_cross_entropy_with_logits(output, target, reduction="none")
    pt = torch.exp(-logpt)

    # compute the loss
    if reduced_threshold is None:
        focal_term = (1.0 - pt).pow(gamma)
    else:
        focal_term = ((1.0 - pt) / reduced_threshold).pow(gamma)
        focal_term[pt < reduced_threshold] = 1

    loss = focal_term * logpt

    if alpha is not None:
        loss *= alpha * target + (1 - alpha) * (1 - target)

    if normalized:
        norm_factor = focal_term.sum().clamp_min(eps)
        loss /= norm_factor

    loss = loss.mean()

    return loss

class FocalLoss(_Loss):

    def __init__(
            self,
            alpha: Optional[float] = None,
            gamma: Optional[float] = 2.,
            ignore_index: Optional[int] = None,
            reduction: Optional[str] = "mean",
            normalized: bool = False,
            reduced_threshold: Optional[float] = None,
    ):
        """Compute Focal loss

        Args:
            mode: Loss mode 'binary', 'multiclass' or 'multilabel'
            alpha: Prior probability of having positive value in target.
            gamma: Power factor for dampening weight (focal strength).
            ignore_index: If not None, targets may contain values to be ignored.
                Target values equal to ignore_index will be ignored from loss computation.
            normalized: Compute normalized focal loss (https://arxiv.org/pdf/1909.07829.pdf).
            reduced_threshold: Switch to reduced focal loss. Note, when using this mode you should use `reduction="sum"`.

        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt

        """
        super().__init__()

        self.ignore_index = ignore_index
        self.focal_loss_fn = partial(
            focal_loss_with_logits,
            alpha=alpha,
            gamma=gamma,
            reduced_threshold=reduced_threshold,
            normalized=normalized,
        )

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        num_classes = y_pred.size(1)
        loss = 0

        # Filter anchors with -1 label from loss computation
        if self.ignore_index is not None:
            not_ignored = y_true != self.ignore_index

        for cls in range(num_classes):
            cls_y_true = (y_true == cls).long()
            cls_y_pred = y_pred[:, cls, ...]

            if self.ignore_index is not None:
                cls_y_true = cls_y_true[not_ignored]
                cls_y_pred = cls_y_pred[not_ignored]

            loss += self.focal_loss_fn(cls_y_pred, cls_y_true)

        return loss


def get_data(mode):
    x, y = [], []
    for i in tqdm.tqdm(os.listdir('data/{}/images'.format(mode))):
        img = cv2.imread('data/{}/images/{}'.format(mode, i))
        mask = np.array(Image.open('data/{}/1st_manual/{}.gif'.format(mode, i.split('.')[0])))

        mask[mask != 0] = 1

        img = cv2.resize(img, (256, 256))
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        img = np.expand_dims(cv2.split(img)[1], axis=-1)

        x.append(img)
        y.append(mask)

    x, y = np.array(x, dtype=np.float) / 255.0, np.array(y)
    x = np.transpose(x, axes=[0, 3, 1, 2])

    print('success load {} set'.format(mode))
    return x, y


def metrice(y_true, y_pred):
    y_true, y_pred = y_true.to('cpu').detach().numpy(), np.argmax(y_pred.to('cpu').detach().numpy(), axis=1)
    y_true, y_pred = y_true.reshape((-1)), y_pred.reshape((-1))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    pa = np.diag(cm).sum() / (cm.sum() + 1e-7)

    mpa_arr = np.diag(cm) / (cm.sum(axis=1) + 1e-7)
    mpa = np.nanmean(mpa_arr)

    MIoU = np.diag(cm) / (np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm) + 1e-7)
    MIoU = np.nanmean(MIoU)

    return pa, mpa, MIoU


if __name__ == '__main__':
    BATCH_SIZE = 8

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(DEVICE)

    Name = 'RepUNet'
    model = RepUNet()
    print(sum(p.numel() for p in model.parameters()))

    x_train, y_train = get_data('train')
    x_val, y_val = get_data('test')

    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_iter = torch.utils.data.DataLoader(train_dataset, BATCH_SIZE, shuffle=True)

    val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
    val_iter = torch.utils.data.DataLoader(val_dataset, BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)
    loss = FocalLoss(alpha=0.25).to(DEVICE)

    with open('{}.log'.format(Name), 'w+') as f:
        f.write('epoch,train_loss,test_loss,train_pa,test_pa,train_mpa,test_mpa,train_miou,test_miou')
    best_miou = 0
    print('{} begin train!'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    for epoch in range(100):
        model.to(DEVICE)
        model.train()
        train_loss = 0
        begin = time.time()
        num = 0
        train_pa, train_mpa, train_miou = 0, 0, 0
        for x, y in train_iter:
            x, y = x.to(DEVICE), y.to(DEVICE).long()
            pred = model(x.float())
            l = loss(pred, y)
            train_loss += float(l.data)

            for module in model.modules():
                if hasattr(module, 'get_custom_L2'):
                    l += 0.5 * module.get_custom_L2() * 0.001

            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            train_pa_, train_mpa_, train_miou_ = metrice(y, pred)
            train_pa += train_pa_
            train_mpa += train_mpa_
            train_miou += train_miou_
            num += 1
        train_loss /= num
        train_pa, train_mpa, train_miou = train_pa / num, train_mpa / num, train_miou / num

        num = 0
        test_loss = 0
        model.eval()
        test_pa, test_mpa, test_miou = 0, 0, 0
        with torch.no_grad():
            for x, y in val_iter:
                x, y = x.to(DEVICE), y.to(DEVICE).long()

                pred = model(x.float())
                l = loss(pred, y)
                num += 1
                test_loss += float(l.data)

                test_pa_, test_mpa_, test_miou_ = metrice(y, pred)
                test_pa += test_pa_
                test_mpa += test_mpa_
                test_miou += test_miou_

        test_loss /= num
        test_pa, test_mpa, test_miou = test_pa / num, test_mpa / num, test_miou / num

        if test_miou > best_miou:
            best_miou = test_miou
            model.to('cpu')
            torch.save(model, '{}.pkl'.format(Name))
        print(
            '{} epoch:{}, time:{:.2f}s, train_loss:{:.4f}, val_loss:{:.4f}, train_pa:{:.4f}, val_pa:{:.4f}, train_mpa:{:.4f}, test_mpa:{:.4f}, train_miou:{:.4f}, test_miou:{:.4f}'.format(
                datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                epoch + 1, time.time() - begin, train_loss, test_loss,
                train_pa, test_pa, train_mpa, test_mpa, train_miou, test_miou
            ))
        with open('{}.log'.format(Name), 'a+') as f:
            f.write('\n{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}'.format(
                epoch, train_loss, test_loss, train_pa, test_pa, train_mpa, test_mpa, train_miou, test_miou
            ))
