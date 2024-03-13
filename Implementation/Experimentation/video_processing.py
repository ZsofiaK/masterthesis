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

def shorten_to_odd(cap, verbose=False):
    '''Shortens a video to the maximum odd number of seconds.'''

    all_frames = get_frames(cap)

    # Get the frames per second (fps) and total number of frames
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    nr_frames = len(all_frames)

    if verbose:
        print(f'FPS: {fps}, TOTAL FRAMES: {nr_frames}')
    
    # Calculate the maximum length in whole seconds.
    max_round_length = int(np.floor(nr_frames / fps))
    
    if verbose:
        print(f'MAX ROUND LENGTH: {max_round_length}')

    # Calculate the maximum length in odd whole seconds.
    max_odd_length = max_round_length - (1 - (max_round_length % 2))

    if verbose:
        print(f'MAX ODD LENGTH: {max_odd_length}')
    
    # Calculate the maximum number of frames
    max_frames = fps * max_odd_length

    if verbose:
        print(f'MAX FRAMES: {max_frames}')
    
    # Calculate the number of frames to discard from the beginning and end
    to_discard = (nr_frames - max_frames) // 2

    if verbose:
        print(f'TO DISCARD: {to_discard}')
    
    # Discard frames from the beginning and end
    shortened_frames = all_frames[to_discard:-to_discard]
    
    # Ensure the length of frames is no more than max_frames
    if len(shortened_frames) > max_frames:
        shortened_frames = shortened_frames[:max_frames]

    return shortened_frames

def separate_to_clips(frames, seconds, fps):
    '''Separates a list of video frames into clips of given length.'''

    nr_frames = len(frames)

    # Number of frames in a clip (rounding down if seconds is not an integer).
    frames_in_a_clip = int(seconds * fps)

    # Checking if any frames will be left over at the end if separating into the given length clips.
    leftovers = nr_frames % (frames_in_a_clip)

    if leftovers > 0:
        print(f'Warning: {leftovers} frames will be discarded.')

    # Create a list of sublists, each containing fps number of frames
    clip_frames = [frames[i:i+frames_in_a_clip] for i in range(0, len(frames), frames_in_a_clip)]
    
    return clip_frames
