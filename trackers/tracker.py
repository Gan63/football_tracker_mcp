import ultralytics
import supervision as sv
import pickle
import numpy as np
import cv2
from ultralytics.models import YOLO


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        
    def detect_frames(self, frame_generator):
        """Detect objects in frames from a generator"""
        batch_size = 20
        detections = []
        frames_batch = []
        
        for frame in frame_generator:
            frames_batch.append(frame)
            if len(frames_batch) == batch_size:
                detections.extend(self.model.predict(frames_batch, conf=0.1))
                frames_batch = []
        if frames_batch: # Process remaining frames
            detections.extend(self.model.predict(frames_batch, conf=0.1))

        return detections
    
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """Get tracked objects from frames"""
        
        if read_from_stub and stub_path:
            try:
                with open(stub_path, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
        
        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }
        
        detections = []
        for frame in frames:
            detections.extend(self.model.predict([frame], conf=0.1))

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}
            
            # Convert to supervision format
            detection_supervision = sv.Detections.from_ultralytics(detection)
            
            # Track objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]
                
                # Use 'person' instead of 'player' for YOLO models
                if cls_id == cls_names_inv.get("person", cls_names_inv.get("player", -1)):
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                
                if cls_id == cls_names_inv.get("referee", -1):
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}
                    
            # Handle ball detection separately
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                
                if cls_id == cls_names_inv.get("ball", cls_names_inv.get("sports ball", -1)):
                    tracks["ball"][frame_num][1] = {"bbox": bbox}
        
        if stub_path:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
        
        return tracks
    
    def draw_annotations(self, frames, tracks, team_ball_possession, specific_frame_num=None):
        """Draw bounding boxes and annotations on frames"""
        output_frames = []
        
        start_frame = specific_frame_num if specific_frame_num is not None else 0
        for i, frame in enumerate(frames):
            frame_num = start_frame + i
            frame = frame.copy()
            
            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            
            # Draw players
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 255, 0))
                bbox = player["bbox"]
                
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"P{track_id}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw referees
            for track_id, referee in referee_dict.items():
                bbox = referee["bbox"]
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            
            # Draw ball
            for track_id, ball in ball_dict.items():
                bbox = ball["bbox"]
                x1, y1, x2, y2 = map(int, bbox)
                cv2.circle(frame, (int((x1+x2)/2), int((y1+y2)/2)), 10, (0, 0, 255), -1)
            
            # Draw possession stats
            if frame_num < len(team_ball_possession):
                possession = team_ball_possession[:frame_num+1]
                team_1_poss = np.sum(np.array(possession) == 1) / len(possession) * 100
                team_2_poss = np.sum(np.array(possession) == 2) / len(possession) * 100
                
                cv2.putText(frame, f"Team 1: {team_1_poss:.1f}%", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
                cv2.putText(frame, f"Team 2: {team_2_poss:.1f}%", (50, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
            
            output_frames.append(frame)
        
        return output_frames
    
    def ball_interpolation(self, ball_positions):
        """Interpolate missing ball positions"""
        ball_positions_filled = [{1: pos[1]} if 1 in pos else {} 
                                for pos in ball_positions]
        
        df_ball = []
        for pos in ball_positions_filled:
            df_ball.append(pos.get(1, {}).get("bbox", [None]*4))
        
        df_ball = np.array(df_ball)
        
        # Simple linear interpolation
        for i in range(len(df_ball)):
            if all(x is None for x in df_ball[i]):
                # Find previous and next valid positions
                prev_idx = i - 1
                while prev_idx >= 0 and all(x is None for x in df_ball[prev_idx]):
                    prev_idx -= 1
                
                next_idx = i + 1
                while next_idx < len(df_ball) and all(x is None for x in df_ball[next_idx]):
                    next_idx += 1
                
                if prev_idx >= 0 and next_idx < len(df_ball):
                    # Interpolate
                    alpha = (i - prev_idx) / (next_idx - prev_idx)
                    df_ball[i] = (1 - alpha) * np.array(df_ball[prev_idx]) + \
                                alpha * np.array(df_ball[next_idx])
        
        # Update ball positions
        for i, bbox in enumerate(df_ball):
            if not all(x is None for x in bbox):
                ball_positions[i][1] = {"bbox": bbox.tolist()}
        
        return ball_positions
    
    def add_position_to_tracks(self, tracks):
        """Add center position to each track"""
        for object_type, object_tracks in tracks.items():
            for frame_num, frame_tracks in enumerate(object_tracks):
                for track_id, track in frame_tracks.items():
                    bbox = track["bbox"]
                    center_x = (bbox[0] + bbox[2]) / 2
                    center_y = (bbox[1] + bbox[3]) / 2
                    tracks[object_type][frame_num][track_id]["position"] = (center_x, center_y)
            