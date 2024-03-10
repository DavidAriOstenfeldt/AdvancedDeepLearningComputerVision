import os
from tqdm import tqdm
from PIL import Image
from queue import Queue
import cv2
import numpy as np


import argparse

# torch imports
import torch
from torchvision import transforms

from model import Network
from util import compute_inbetween_frames

VIDEO = 'soapbox'

parser = argparse.ArgumentParser()
parser.add_argument('--conf', type=int, default=1, help='Configuration for losses (1-3)')
args = parser.parse_args()
conf = args.conf
assert conf in {1,2,3}

video_path = f'./data/DAVIS17/JPEGImages/480p/{VIDEO}'
output_path = f'results/{VIDEO}/{conf}'
frames_path = os.path.join(output_path, 'predicted_frames')
os.makedirs(output_path, exist_ok=True)
os.makedirs(frames_path, exist_ok=True)

n_frames = len(os.listdir(video_path))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = Network()
model.to(device)
model.eval()
model.load_state_dict(torch.load(os.path.join(f'weights/conf-{conf}.pth')))

frame_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


real_frames = Queue()
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
video_writer = cv2.VideoWriter(os.path.join(output_path, 'slow_motion_video.mp4'), fourcc, 10, (128,128))
orig_video_writer = cv2.VideoWriter(os.path.join(output_path, 'original_video.mp4'), fourcc, 5, (128,128))

for frame_num in tqdm(range(n_frames), desc=f'conf-{conf}'):
    frame = Image.open(os.path.join(video_path, f'{frame_num:05d}.jpg')).convert('RGB')
    frame = frame.resize((128,128))
    orig_video_writer.write(np.array(frame)[:, :, ::-1])

    frame = frame_transform(frame).unsqueeze(0) # batch dim
    frame = frame.to(device)

    real_frames.put((frame_num, frame))

    if real_frames.qsize() == 4:
        #Compute the intermediate frames between the two middle frames in the queue
        compute_inbetween_frames(model,real_frames, 1, video_writer, frames_path)

        #In the very beginning and end of our video, we need to predict the first and last frame pair, respectively
        if real_frames.queue[0][0] == 0:
            compute_inbetween_frames(model,real_frames, 0, video_writer, frames_path)
        if real_frames.queue[3][0] == n_frames-1:
            compute_inbetween_frames(model,real_frames, 2, video_writer, frames_path)
        real_frames.get()

video_writer.release()
orig_video_writer.release()
