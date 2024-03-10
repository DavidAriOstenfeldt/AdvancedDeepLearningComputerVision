import os
import torch
from PIL import Image
import cv2

from dataset.helpers import *

def write_frame(frame, frame_no, video_writer, frames_path):
    im_np = im_normalize(tens2image(frame.cpu().detach()))
    video_writer.write(cv2.cvtColor(np.uint8(im_np*255), cv2.COLOR_RGB2BGR))
    
    if frames_path is not None:
        out_path = os.path.join(frames_path , f'{frame_no:05d}.png')
        im = Image.fromarray(np.uint8(im_np*255))
        im.save(out_path)
    


def recursive_predict_and_write(model,f0, f2, frame0_no, frame_diff, video_writer, frames_path):
    #Interpolate the middle frame
    with torch.no_grad():
        f1 = model(f0, f2)['output_im']
    offset = frame_diff//2
    #Recursively predict the extra frames
    if frame_diff > 2:
        recursive_predict_and_write(f0, f1, frame0_no, offset)
        recursive_predict_and_write(f1, f2, frame0_no+offset, offset)
    
    f0_np = im_normalize(tens2image(f0.cpu().detach()))
    video_writer.write(cv2.cvtColor(np.uint8(f0_np*255), cv2.COLOR_RGB2BGR))

    write_frame(f1, frame0_no+offset, video_writer, frames_path)
          


def compute_inbetween_frames(model,real_frames, idx, video_writer, frames_path, slow_factor=2):
    frame0_no, frame0 = real_frames.queue[idx]
    frame1_no, frame1 = real_frames.queue[idx+1]
    recursive_predict_and_write(model, frame0, frame1, frame0_no*slow_factor, slow_factor, video_writer, frames_path)