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

def shorten_to_odd(cap):
    '''Shortens a video to the maximum odd number of seconds.'''

    all_frames = get_frames(cap)

    # Get the frames per second (fps) and total number of frames
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    nr_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate the maximum odd length in seconds
    max_odd_length = int(np.floor(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000))
    
    # Calculate the maximum number of frames
    max_frames = fps * max_odd_length
    
    # Calculate the number of frames to discard from the beginning and end
    to_discard = (nr_frames - max_frames) // 2
    
    # Discard frames from the beginning and end
    shortened_frames = all_frames[to_discard:-to_discard]
    
    # Ensure the length of frames is no more than max_frames
    if len(shortened_frames) > max_frames:
        shortened_frames = shortened_frames[:max_frames]

    return shortened_frames

def separate_to_clips(frames, seconds, fps):
    '''Separates a list of video frames into clips of given length.'''

    # Create a list of sublists, each containing fps number of frames
    clip_frames = [frames[i:i+fps] for i in range(0, len(frames), seconds*fps)]
    
    return clip_frames