import os
import cv2
import gc
import numpy as np

# Disable YOLO internet calls (IMPORTANT for Render)
os.environ["YOLO_OFFLINE"] = "1"
os.environ["YOLO_NO_VERIFY"] = "1"
os.environ["ULTRALYTICS_HUB"] = "0"
os.environ["YOLO_DISABLE_UPDATE"] = "1"

from trackers import *
from team_assigner import *
from player_ball_assigner import *
from camera_movement import *
from view_transformation import *
from speed_and_distance import *
from ultralytics import YOLO

os.environ["LOKY_MAX_CPU_COUNT"] = "4"


def process_video_optimized(input_path, output_path):

    try:
        print("Reading video...")
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError("Could not open input video")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # --- Use AVI instead of MP4 (Render safe)
        output_path = output_path.replace(".mp4", ".avi")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print("Loading lightweight YOLO model (yolov8n.pt)...")
        model = YOLO("models/yolov8n.pt")      # <-- LOCAL MODEL ONLY
        tracker = Tracker(model)

        team_assigner = TeamAssigner()
        player_assigner = PlayerBallAssigner()
        speed_est = SpeedAndDistance_Estimator()
        camera_movement = None

        team_colors_assigned = False
        team_ball_possession = []

        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if camera_movement is None:
                camera_movement = CameraMovement(frame)

            frame_tracks = tracker.get_object_tracks([frame], read_from_stub=False)
            tracker.add_position_to_tracks(frame_tracks)

            cam_shift = camera_movement.get_camera_movement([frame])
            camera_movement.adjust_tracks_positions(frame_tracks, cam_shift)

            players_dict = frame_tracks.get("players", {})
            if (not team_colors_assigned) and len(players_dict) > 0:
                first_player_set = list(players_dict.values())[0]
                team_assigner.assign_team_color(frame, first_player_set)
                team_colors_assigned = True

            for pid, pdata in players_dict.items():
                team = team_assigner.get_player_team(frame, pdata["bbox"], pid)
                pdata["team"] = team
                pdata["team_color"] = team_assigner.team_colors.get(team, [255, 255, 255])

            ball_dict = frame_tracks.get("ball", {})
            ball_bbox = ball_dict.get(1, {}).get("bbox")

            if ball_bbox:
                nearest = player_assigner.assign_ball_to_player(players_dict, ball_bbox)
                if nearest != -1:
                    players_dict[nearest]["ball_possession"] = True
                    team_ball_possession.append(players_dict[nearest]["team"])
                else:
                    team_ball_possession.append(team_ball_possession[-1] if team_ball_possession else 1)
            else:
                team_ball_possession.append(team_ball_possession[-1] if team_ball_possession else 1)

            speed_est.add_speed_and_distance(frame_tracks)

            annotated = tracker.draw_annotations([frame], frame_tracks, np.array(team_ball_possession))[0]
            annotated = camera_movement.draw_camera_movement(annotated, cam_shift)
            annotated = speed_est.draw_speed_and_distance(annotated, frame_tracks)

            out.write(annotated)
            frame_id += 1

        cap.release()
        out.release()

        return {
            "processed_video_url": os.path.basename(output_path)
        }

    except Exception as e:
        print(f"Processing error: {e}")
        return {"error": str(e)}

    finally:
        gc.collect()
