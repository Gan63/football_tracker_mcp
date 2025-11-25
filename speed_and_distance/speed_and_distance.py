import cv2
import numpy as np

class SpeedAndDistance_Estimator:
    def __init__(self, frame_rate=24):
        self.frame_window = 5
        self.frame_rate = frame_rate
        self.pixel_to_meter_ratio = 0.05 # Example: 1 pixel = 5cm
    
    def add_speed_and_distance(self, tracks):
        """Calculate speed and distance for each player"""
        
        total_distances = {}
        previous_positions = {}
        
        for object_type, object_tracks in tracks.items():
            if object_type == "ball" or object_type == "referees":
                continue
            
            for frame_num, frame_tracks in enumerate(object_tracks):
                for track_id, track_info in frame_tracks.items():
                    position = track_info.get("position_adjusted", track_info.get("position"))
                    if position is None:
                        continue
                    
                    if track_id in previous_positions:
                        prev_pos = previous_positions[track_id]
                        distance_pixels = np.linalg.norm(np.array(position) - np.array(prev_pos))
                        distance_meters = distance_pixels * self.pixel_to_meter_ratio
                        
                        if track_id not in total_distances:
                            total_distances[track_id] = 0
                        total_distances[track_id] += distance_meters
                        
                        # Speed calculation over a window for stability
                        time_elapsed = 1 / self.frame_rate
                        speed_ms = distance_meters / time_elapsed if time_elapsed > 0 else 0
                        speed_kmh = speed_ms * 3.6
                        
                        # Use a running average for speed to smooth it out
                        if "speed" in track_info:
                            track_info["speed"] = (track_info["speed"] + speed_kmh) / 2
                        else:
                            track_info["speed"] = speed_kmh
                    
                    previous_positions[track_id] = position
                    track_info["distance"] = total_distances.get(track_id, 0)
    
    def draw_speed_and_distance(self, frames, tracks, specific_frame_num=None):
        """Draw speed and distance information on frames"""
        output_frames = []
        
        start_frame = specific_frame_num if specific_frame_num is not None else 0
        for i, frame in enumerate(frames):
            frame_num = start_frame + i
            frame = frame.copy()
            
            player_dict = tracks["players"][frame_num]
            
            for track_id, player in player_dict.items():
                speed = player.get("speed", 0)
                distance = player.get("distance", 0)
                
                if "bbox" in player:
                    bbox = player["bbox"]
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    cv2.putText(frame, f"S:{speed:.1f}", (x1, y2+20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    cv2.putText(frame, f"D:{distance:.1f}", (x1, y2+35),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            output_frames.append(frame)
        
        return output_frames