from sklearn.cluster import KMeans
import cv2
import numpy as np

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
        self.kmeans = None
        
    def get_clustering_model(self, image):
        """Create K-means clustering model for team colors"""
        image_2d = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10)
        kmeans.fit(image_2d)
        return kmeans
    
    def get_player_color(self, frame, bbox):
        """Extract dominant color from player bbox"""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Get upper half of bbox (jersey area)
        player_img = frame[y1:y2, x1:x2]
        top_half = player_img[0:int(player_img.shape[0]/2), :]
        
        if top_half.size == 0:
            return None
        
        # Get dominant color
        kmeans = self.get_clustering_model(top_half)
        cluster_labels = kmeans.labels_
        
        # Get most common cluster
        clustered_img = cluster_labels.reshape(top_half.shape[0], top_half.shape[1])
        corner_clusters = [clustered_img[0, 0], clustered_img[0, -1],
                          clustered_img[-1, 0], clustered_img[-1, -1]]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster
        
        player_color = kmeans.cluster_centers_[player_cluster]
        return player_color
    
    def assign_team_color(self, frame, player_detections):
        """Assign team colors based on first frame"""
        player_colors = []
        
        for track_id, player in player_detections.items():
            bbox = player["bbox"]
            color = self.get_player_color(frame, bbox)
            if color is not None:
                player_colors.append(color)
        
        if len(player_colors) < 2:
            self.team_colors[1] = np.array([255, 0, 0])
            self.team_colors[2] = np.array([0, 0, 255])
            return
        
        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10)
        kmeans.fit(player_colors)
        
        self.kmeans = kmeans
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]
    
    def get_player_team(self, frame, bbox, player_id):
        """Get team assignment for a player"""
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        player_color = self.get_player_color(frame, bbox)
        
        if player_color is None:
            return 1
        
        if self.kmeans is None:
            return 1
        
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id = int(team_id) + 1
        
        self.player_team_dict[player_id] = team_id
        
        return team_id