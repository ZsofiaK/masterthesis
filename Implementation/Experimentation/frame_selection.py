from typing import List
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def count_frames(video_path):
  '''
  Count the number of frames in a video.
  '''

  cap = cv2.VideoCapture(video_path)

  frame_count = 0

  while True:
      ret, frame = cap.read()

      if not ret:
          break

      frame_count += 1

  cap.release()

  return frame_count

def calculate_ssim(frame1, frame2):
    """Calculate Structural Similarity Index (SSIM) between two frames."""

    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    ssim_score, _ = ssim(gray1, gray2, full=True)

    return ssim_score

def select_frames_ssim(video_path, nr_frames_to_select=10, max_ssim=0.95):
    """
    Select keyframes based on scene change detection using SSIM.
    """

    cap = cv2.VideoCapture(video_path)

    # Check if the video capture object is opened successfully
    if not cap.isOpened():
        print(f"Error: Unable to open video file '{video_path}'")
        return None, None

    # Read the first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Unable to read the first frame.")
        cap.release()
        return None, None

    keyframes = [prev_frame]
    selected_indices = [0]  # Index of the first frame

    # Starting from the second frame.
    frame_index = 1

    while True:
        # Read the next frame.
        ret, curr_frame = cap.read()

        # Breaking loop if unable to read frame.
        if not ret:
            print(f"Warning: Unable to read frame at index {frame_index}. Breaking the loop.")
            break

        # Calculate SSIM between consecutive frames
        ssim_score = calculate_ssim(prev_frame, curr_frame)

        # Check if calculate_ssim is returning a valid value
        if ssim_score is None:
            print(f"Error: calculate_ssim returned None for frame at index {frame_index}.")
            cap.release()
            return None, None

        # Select current frame as a keyframe if it is different from the previous one.
        if np.any(ssim_score < max_ssim):
            keyframes.append(curr_frame)
            selected_indices.append(frame_index)

        # Stop when the desired number of keyframes is reached.
        if len(keyframes) == nr_frames_to_select:
            break

        prev_frame = curr_frame
        frame_index += 1

    cap.release()
    return keyframes, selected_indices

def select_frames_evenly(video_path:str, total_frames:int, num_frames:int=10):
    """
    Select evenly spaced frames from a video file.
    
    :param: video_path: Path to the video file as string.
    :param: num_frames: Number of frames to select (integer).
    :return: list of selected frame indices.
    """
    
    selected_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
    
    
    return [item[0] for item in selected_indices]
