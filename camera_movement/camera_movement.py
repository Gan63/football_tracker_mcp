import cv2
import numpy as np
import pickle

class CameraMovement:
    def __init__(self, frame):
        self.min_distance = 5
        
        # Parameters for corner detection
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        first_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(first_gray)
        mask[:, 0:20] = 1
        mask[:, frame.shape[1]-20:] = 1
        
        self.features = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7,
            mask=mask
        )
    
    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        """Calculate camera movement between frames"""
        
        if read_from_stub and stub_path:
            try:
                with open(stub_path, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
        
        frame_list = list(frames)
        if not frame_list:
            return []
        
        camera_movement = [[0, 0]] * len(frame_list)
        
        old_gray = cv2.cvtColor(frame_list[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)
        
        for frame_num in range(1, len(frame_list)):
            frame_gray = cv2.cvtColor(frame_list[frame_num], cv2.COLOR_BGR2GRAY)
            
            if old_features is not None and len(old_features) > 0:
                new_features, status, error = cv2.calcOpticalFlowPyrLK(
                    old_gray, frame_gray, old_features, None, **self.lk_params,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
                )
                
                if new_features is not None:
                    # Calculate movement
                    max_distance = 0
                    camera_movement_x, camera_movement_y = 0, 0
                    
                    for i, (new, old) in enumerate(zip(new_features, old_features)):
                        if status[i]:
                            new_pos = new.ravel()
                            old_pos = old.ravel()
                            
                            distance = np.linalg.norm(new_pos - old_pos)
                            
                            if distance > max_distance:
                                max_distance = distance
                                camera_movement_x = new_pos[0] - old_pos[0]
                                camera_movement_y = new_pos[1] - old_pos[1]
                    
                    if max_distance > self.min_distance:
                        camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                        old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
                    
                    old_gray = frame_gray.copy()
        
        if stub_path:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)
        
        return camera_movement
    
    def adjust__tracks_positions(self, tracks, camera_movement):
        """Adjust track positions based on camera movement"""
        for object_type, object_tracks in tracks.items():
            for frame_num, frame_tracks in enumerate(object_tracks):
                for track_id, track in frame_tracks.items():
                    position = track.get("position", [0, 0])
                    camera_adjustment = camera_movement[frame_num]
                    
                    adjusted_position = (
                        position[0] - camera_adjustment[0],
                        position[1] - camera_adjustment[1]
                    )
                    
                    tracks[object_type][frame_num][track_id]["position_adjusted"] = adjusted_position
    
    def adjust_single_frame_tracks(self, tracks, frame_num, camera_adjustment):
        """Adjust track positions for a single frame"""
        for object_type, object_tracks in tracks.items():
            if frame_num < len(object_tracks):
                for track_id, track in object_tracks[frame_num].items():
                    position = track.get("position", (0, 0))
                    adjusted_position = (
                        position[0] - camera_adjustment[0],
                        position[1] - camera_adjustment[1]
                    )
                    tracks[object_type][frame_num][track_id]["position_adjusted"] = adjusted_position

    def draw_camera_movement(self, frames, camera_movement):
        """Draw camera movement on frames"""
        output_frames = []
        
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()
            
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)
            alpha = 0.6
            frame = cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)
            
            x_movement, y_movement = camera_movement[frame_num]
            
            cv2.putText(frame, f"Camera Movement X: {x_movement:.2f}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(frame, f"Camera Movement Y: {y_movement:.2f}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            output_frames.append(frame)
        
        return output_frames