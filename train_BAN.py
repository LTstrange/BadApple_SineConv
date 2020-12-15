import torch
import pickle
import cv2
import numpy as np
from torch import optim, nn
import time

print('读取数据中...')
with open('data/video720p-half.array', 'rb') as file:
    video = pickle.load(file)
    video = video / 255

print('读取完毕')
# test
print('数据预处理...')
video = np.array([cv2.resize(frame.reshape(540, 720), (512, 512)) for frame in video])
video = video.reshape((-1, 1, 512, 512))
trainloader = torch.utils.data.DataLoader(video, batch_size=32, shuffle=True, pin_memory=True)
print("数据集准备完毕")

from model import AutoEncoder, BAN_decoder, Sine_decoder

autoencoder = AutoEncoder().cuda()
sin_decoder = Sine_decoder().cuda()
ban_decoder = BAN_decoder().cuda()

autoencoder.decoder = ban_decoder

criterion = nn.MSELoss().cuda()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

print("开始训练 training GeneralConv...")
stime = time.time()
for epoch in range(20):
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
            print('[%2d, %5d] loss: %.4f training GeneralConv...' %
                  (epoch + 1, i + 1, running_loss / 50))

            running_loss = 0.0
print(f'花费时间：{time.time() - stime: .2f}s')

with open('module.net', 'wb') as file:
    pickle.dump(autoencoder.cpu(), file)
