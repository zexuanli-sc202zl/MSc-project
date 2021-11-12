import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('RepUNet.log')
epoch = data['epoch']
pa = data['train_pa']
mpa = data['train_mpa']
miou = data['train_miou']
loss = data['train_loss']
val_pa = data['test_pa']
val_mpa = data['test_mpa']
val_miou = data['test_miou']
val_loss = data['test_loss']

plt.figure(figsize=(8, 8))

plt.subplot(2, 2, 1)
plt.plot(epoch, loss, label='train')
plt.plot(epoch, val_loss, label='test')
plt.legend()
plt.ylabel('loss')
plt.xlabel('epoch')

plt.subplot(2, 2, 2)
plt.plot(epoch, pa, label='train')
plt.plot(epoch, val_pa, label='test')
plt.legend()
plt.ylabel('pa')
plt.xlabel('epoch')

plt.subplot(2, 2, 3)
plt.plot(epoch, mpa, label='train')
plt.plot(epoch, val_mpa, label='test')
plt.legend()
plt.ylabel('mpa')
plt.xlabel('epoch')

plt.subplot(2, 2, 4)
plt.plot(epoch, miou, label='train')
plt.plot(epoch, val_miou, label='test')
plt.legend()
plt.ylabel('miou')
plt.xlabel('epoch')

plt.show()