import os
import cv2 
import random
from tqdm import tqdm

random.seed(29) # NOTE: DONT CHANGE!

REGION_SIZE = (128, 128)

frames_path = 'data/DAVIS17/JPEGImages/480p/'
with open(os.path.join('data/DAVIS17/ImageSets/2017/train.txt')) as fp:
    train_vidoes = [line.rstrip('\n') for line in fp]

with open(os.path.join('data/DAVIS17/ImageSets/2017/val.txt')) as fp:
    val_vidoes = [line.rstrip('\n') for line in fp]

davis_triplets_path = 'data/DAVIS_Triplets'
os.makedirs(davis_triplets_path, exist_ok=True)

train_cnt = -1
val_cnt = -1
for video in tqdm(sorted(os.listdir(frames_path)), desc='Creating triplets'):
    
    num_frames = len(os.listdir(os.path.join(frames_path, video)))
    total_frames = num_frames - (num_frames % 3)
    frame_triplets = [(i, i+1, i+2) for i in range(0, total_frames, 3)]
    
    for triplet in frame_triplets:
        f1, f2, f3 = triplet
        f1 = cv2.imread(os.path.join(frames_path, video, f'{f1:05d}.jpg'))
        f2 = cv2.imread(os.path.join(frames_path, video, f'{f2:05d}.jpg'))
        f3 = cv2.imread(os.path.join(frames_path, video, f'{f3:05d}.jpg'))

        frame_height, frame_width, _ = f1.shape

        start_x = random.randint(0, frame_width - REGION_SIZE[0])
        start_y = random.randint(0, frame_height - REGION_SIZE[1])

        f1_region = f1[start_y:start_y + REGION_SIZE[1], start_x:start_x + REGION_SIZE[0]]
        f2_region = f2[start_y:start_y + REGION_SIZE[1], start_x:start_x + REGION_SIZE[0]]
        f3_region = f3[start_y:start_y + REGION_SIZE[1], start_x:start_x + REGION_SIZE[0]]

        if video in train_vidoes:
            train_cnt += 1
            this_out_path = os.path.join(davis_triplets_path, 'train',f'triplet_{train_cnt:06d}')
        else:
            val_cnt += 1
            this_out_path = os.path.join(davis_triplets_path, 'val',f'triplet_{val_cnt:06d}')

        os.makedirs(this_out_path, exist_ok=True)
        
        cv2.imwrite(os.path.join(this_out_path, '1.png'), f1_region)
        cv2.imwrite(os.path.join(this_out_path, '2.png'), f2_region)
        cv2.imwrite(os.path.join(this_out_path, '3.png'), f3_region)
        
