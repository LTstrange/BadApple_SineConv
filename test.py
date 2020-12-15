import torch
import pickle
import cv2
import numpy as np

print('读取文件中...')
with open('module_sine_decoder.net', 'rb') as file:
    autoencoder = pickle.load(file).cuda().requires_grad_(False)

with open('data/video720p-half.array', 'rb') as file:
    video = pickle.load(file)
    video = video / 255


DCVideo = []
print('转换中...')
for frame, image in enumerate(video):
    image = cv2.resize(image[0], (512, 512)).reshape(1, 1, 512, 512)
    image = torch.tensor(image, dtype=torch.float).cuda()
    _, decode = autoencoder(image)

    DCVideo.append(decode.view(512, 512).detach().cpu().numpy())
    if frame % 100 == 0:
        print(f'\r{frame * 100 / 6575: .2f} %', end='')

print(f'\r{(frame+1) * 100 / 6575: .2f} %')

print('保存视频中...')
output = cv2.VideoWriter('result.mp4', cv2.VideoWriter_fourcc(*'MP4V'), int(30), (int(720), int(540)))
for i, image in enumerate(DCVideo):
    result = cv2.resize(image, (720, 540), interpolation=cv2.INTER_LINEAR)
    result = np.array(result * 255, dtype='uint8')
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    output.write(result)
    if i % 100 == 0:
        print(f'\r{i * 100 / 6575: .2f} %', end='')
output.release()
cv2.destroyAllWindows()
print(f'\r{(i+1) * 100 / 6575: .2f} %')

print('开始播放')
for i in range(len(video)):
    cv2.imshow('GT', video[i].reshape(540, 720))

    result = cv2.resize(DCVideo[i], (720, 540), interpolation=cv2.INTER_LINEAR)
    cv2.imshow('decode', result)

    cv2.waitKey(10)



