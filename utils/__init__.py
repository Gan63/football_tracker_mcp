import cv2
import os

def read_video_frames(video_path, start_frame=0):
    """
    Read video frames from a file using a generator, starting from a specific frame.
    This avoids loading the entire video into memory.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
    finally:
        cap.release()

def save_video(frames, output_path, fps=24.0):
    """Save frames to video file"""
    if len(frames) == 0:
        print("No frames to save")
        return

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    height, width = frames[0].shape[:2]

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()
    print(f"Video saved to {output_path}")
