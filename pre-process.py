import numpy as np
import cv2
import pickle

BadApple = cv2.VideoCapture('video/video.mp4')

count = 0
video = []
while True:
    ret, frame = BadApple.read()

    if not ret:
        break
    width, height = BadApple.get(cv2.CAP_PROP_FRAME_WIDTH), BadApple.get(cv2.CAP_PROP_FRAME_HEIGHT)

    frame = cv2.resize(frame, (int(width/2), int(height/2)), interpolation=cv2.INTER_LINEAR)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame = frame.reshape((1, 540, 720))

    video.append(frame)

    if count % 100 == 0:
        print(f'\r{count *100 / 6575: .2f}%', end='')
    count += 1

BadApple.release()
cv2.destroyAllWindows()

print(f'\r{count *100 / 6575: .2f}%')

video = np.array(video)

video = video[:len(video) // 2]

with open(f'data/video{video.shape[-1]}p-half.array', 'wb') as file:
    pickle.dump(video, file)


