from video_processing import read_video, save_video
from trackers import *
from team_assigner import *
from player_ball_assigner import *
from camera_movement import *
from view_transformation import *
from speed_and_distance import *
import numpy as np
import cv2
import os
import gc

os.environ["LOKY_MAX_CPU_COUNT"] = "4"

def process_video_optimized(input_path,output_path):
    """
    Process a football video with player tracking, team assignment, and analysis.
    
    Args:
        input_path (str): Path to input video file
        output_path (str): Path where processed video will be saved
        
    Returns:
        str: Path to the processed video file
        
    Raises:
        ValueError: If video cannot be read
        Exception: If processing fails at any stage
    """
    cap = None
    out = None
    tracker = None
    
    try:
        print("Reading video...")
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError("Error: Could not open video file.")

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        print("Initializing tracker...")
        tracker = Tracker('models/yolov5su/best.pt')

        team_assigner = TeamAssigner()
        player_assigner = PlayerBallAssigner()
        camera_movement = None
        speed_dist = SpeedAndDistance_Estimator()

        tracks = {"players": [], "ball": []}
        team_ball_possession = []

        frame_num = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if camera_movement is None:
                camera_movement = CameraMovement(frame)

            print(f"Processing frame {frame_num}...")

            frame_tracks = tracker.get_object_tracks([frame], read_from_stub=False)
            tracker.add_position_to_tracks(frame_tracks)

            camera_movement_per_frame = camera_movement.get_camera_movement([frame])
            camera_movement.adjust_tracks_positions(frame_tracks, camera_movement_per_frame)

            if "ball" in frame_tracks:
                frame_tracks["ball"] = tracker.ball_interpolation(frame_tracks["ball"])

            speed_dist.add_speed_and_distance(frame_tracks)

            if not team_assigner.team_colors:
                team_assigner.assign_team_color(frame, frame_tracks.get("players", [{}])[0])

            for player_id, p_track in frame_tracks.get("players", [{}])[0].items():
                team = team_assigner.get_player_team(frame, p_track["bbox"], player_id)
                p_track["team"] = team
                p_track["team_color"] = team_assigner.team_colors.get(team, [255, 255, 255])

            ball_bbox = frame_tracks.get("ball", [{}])[0].get(1, {}).get("bbox")
            if ball_bbox is not None:
                closest_player = player_assigner.assign_ball_to_player(frame_tracks["players"][0], ball_bbox)
                if closest_player != -1:
                    frame_tracks["players"][0][closest_player]["ball_possession"] = True
                    team_ball_possession.append(frame_tracks["players"][0][closest_player]["team"])
                else:
                    team_ball_possession.append(team_ball_possession[-1] if team_ball_possession else 1)
            else:
                team_ball_possession.append(team_ball_possession[-1] if team_ball_possession else 1)

            annotated_frame = tracker.draw_annotations([frame], frame_tracks, np.array(team_ball_possession))
            annotated_frame = camera_movement.draw_camera_movement(annotated_frame, camera_movement_per_frame)
            annotated_frame = speed_dist.draw_speed_and_distance(annotated_frame, frame_tracks)

            out.write(annotated_frame[0])
            frame_num += 1

        print("Done!")
        return output_path
        
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        raise

    finally:
        if cap is not None:
            cap.release()
        if out is not None:
            out.release()
        gc.collect()


# Only run this if the script is executed directly (not imported)
if __name__ == '__main__':
    # Example usage - uncomment to test directly

    process_video("input_videos/video2.mp4", "output/output_video.avi")
