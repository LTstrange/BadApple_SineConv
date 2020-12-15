import cv2
import pickle
import numpy as np
import torch
from torch import nn
import time

from model import AutoEncoder, Sine_decoder


print('读取数据中...')
with open('data/video720p-half.array', 'rb') as file:
    video = pickle.load(file)
    video = video / 255

with open('module.net', 'rb') as file:
    autoencoder = pickle.load(file)

print('读取模型与数据完毕')
# test
print('数据预处理...')
video = np.array([cv2.resize(frame.reshape(540, 720), (512, 512)) for frame in video])
video = video.reshape((-1, 1, 512, 512))
trainloader = torch.utils.data.DataLoader(video, batch_size=32, shuffle=True, pin_memory=True)
print("数据集准备完毕")

autoencoder = autoencoder.cuda()
autoencoder.encoder.requires_grad_(False)
autoencoder.decoder = Sine_decoder().requires_grad_(True).cuda()

optimizer = torch.optim.Adam(lr=1e-4, params=autoencoder.parameters())
criterion = nn.MSELoss().cuda()

print("开始训练 training SineConv...")
stime = time.time()
for epoch in range(90):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        optimizer.zero_grad()
        Tensordata = data.float().cuda()
        _, output = autoencoder(Tensordata)
        loss = criterion(output, Tensordata)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 50 == 49:
            print('[%2d, %5d] loss: %.4f training SineConv...' %
                  (epoch + 1, i + 1, running_loss / 50))

            running_loss = 0.0
print(f'花费时间：{time.time() - stime: .2f}s')

with open('module_sine_decoder.net', 'wb') as file:
    pickle.dump(autoencoder.cpu(), file)


