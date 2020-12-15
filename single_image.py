import cv2
import pickle
import torch
from torch import nn
import matplotlib.pyplot as plt

from model import AutoEncoder, Sine_decoder

print('loading...')
with open('data/image720p.array', 'rb') as file:
    origin = pickle.load(file)

with open('module.net', 'rb') as file:
    autoencoder = pickle.load(file)

origin = cv2.resize(origin[0], (512, 512))
image = torch.tensor(origin, dtype=torch.float).view(1, 1, 512, 512).cuda()


autoencoder = autoencoder.cuda()
autoencoder.encoder.requires_grad_(False)
autoencoder.decoder = Sine_decoder().requires_grad_(True).cuda()

optim = torch.optim.Adam(lr=1e-4, params=autoencoder.parameters())
criterion = nn.MSELoss().cuda()


plt.figure()
print('training...')
running_loss = 0.0002
for step in range(3000):
    _, decode = autoencoder(image)

    loss = criterion(decode, image)

    running_loss += loss.item()
    if step % 100 == 99:
        print(f'{running_loss / 100: .4f}, {step+1: 4d}')
        plt.imshow(decode.view(512, 512).detach().cpu().numpy(), cmap='gray')
        plt.title(f'{running_loss / 100: .4f}, {step+1: 4d}')
        plt.show()
        running_loss = 0.0

    optim.zero_grad()
    loss.backward()
    optim.step()
    step += 1



plt.imshow(origin)
plt.show()
# plt.imshow(result.view(512, 512).detach().numpy(), cmap='gray')
# plt.show()
print('finish')




