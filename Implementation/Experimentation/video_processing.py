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

def get_clips(video_path:str, clip_length:float):
    '''Separates a video into clips of equal lengths.
    Restriction: the middle of the video always coincides with the middle of a clip.
    :param: video_path: path to the video.
    :param: clip_length: length of clips in seconds.
    :return: list of clips, where each clip is a list of numpy arrays (i.e., frames).
    '''
    
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_per_clip = int(clip_length * fps)
    
    # Ensure there is at least one frame per clip
    if frames_per_clip < 1:
        raise ValueError("Clip length too short for video's framerate.")
    
    # Calculate number of clips
    total_clips = total_frames // frames_per_clip
    if total_clips % 2 == 0:
        total_clips -= 1  # Ensure there is a middle clip
    
    # Calculate the start frame for each clip
    start_frame_middle_clip = (total_frames // 2) - (frames_per_clip // 2)
    clips_to_side = (total_clips - 1) // 2  # Clips on each side of the middle
    
    clip_starts = [start_frame_middle_clip - (frames_per_clip * i) for i in range(clips_to_side, 0, -1)]
    clip_starts.append(start_frame_middle_clip)
    clip_starts.extend(start_frame_middle_clip + (frames_per_clip * (i + 1)) for i in range(clips_to_side))
    
    # Read and return the clips
    clips = []
    for start in clip_starts:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        clip = [cap.read()[1] for _ in range(frames_per_clip) if cap.isOpened()]
        clips.append(clip)
    
    cap.release()
    
    return clips, clip_starts, fps
