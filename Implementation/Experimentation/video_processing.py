import cv2
import numpy as np

def get_frames(cap):
    # Initialize a list to store frames
    frames = []
    
    # Read frames until the end of the video
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    return frames

def separate_to_clips(frames, seconds, fps):
  '''Separates a list of video frames into clips of given length.
    Importantly, there is always a clip centered around the middle of the video.
    '''

  nr_frames = len(frames)

  # Number of frames in a clip (rounding down if seconds is not an integer).
  frames_in_a_clip = int(seconds * fps)

  first_frame_middle_clip = (nr_frames // 2) - (frames_in_a_clip // 2)

  leftovers_at_beginning = first_frame_middle_clip % frames_in_a_clip

  if leftovers_at_beginning > 0:
      print(f'Warning: {leftovers_at_beginning} frames will be discarded at the beginning of video.')

  leftovers_at_end = (nr_frames - first_frame_middle_clip) % frames_in_a_clip

  if leftovers_at_end > 0:
      print(f'Warning: {leftovers_at_end} frames will be discarded at the end of video.')

  # Create a list of sublists, each containing frames_in_a_clip number of frames.
  clip_frames = []
  
  for i in range(leftovers_at_beginning, nr_frames, frames_in_a_clip):
    if i + frames_in_a_clip <= nr_frames - leftovers_at_end:
      clip_frames.append(frames[i:i+frames_in_a_clip])

  print()

  return clip_frames
