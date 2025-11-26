while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 🟢 LOW-MEMORY FIX — RESIZE FRAME
    frame = cv2.resize(frame, (640, 360))

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

    # 🟩 MEMORY SAVERS — CRITICAL FOR RENDER
    cv2.waitKey(1)
    del frame_tracks
    gc.collect()
