import os
import sys
import time
import math
from collections import defaultdict, deque

import cv2
import mediapipe as mp
from ultralytics import YOLO


# ============================================================
# SETTINGS
# ============================================================

YOLO_MODEL_PATH = "yolo26n.pt"
POSE_MODEL_PATH = "pose_landmarker_heavy.task"

# ============================================================
# CAMERA SETTINGS — macOS + iPhone
# ============================================================

# 0 = usually the MacBook camera.
# Other indexes may correspond to your iPhone.
# The correct index depends on how macOS lists your cameras.
CAMERA_INDEX = 1

# Use the native macOS camera backend.
CAMERA_BACKEND = cv2.CAP_AVFOUNDATION

# Mirror the camera image horizontally.
# Note: macOS AVFoundation / Continuity Camera may already be mirrored by default.
# If your camera is already mirrored by macOS, flipping it will un-mirror it.
# You can also press 'M' in the preview window to toggle mirror mode live.
MIRROR_CAMERA = False


# Scale factor for the camera frame (0.5 for 0.5x scale / half resolution).
FRAME_SCALE = 0.5

MAX_POSES = 5

# Movement analysis
HISTORY_SECONDS = 5.0
MOVEMENT_WINDOW_SECONDS = 1.5
SPEED_SMOOTHING_SAMPLES = 5

MIN_MOVEMENT_METERS = 0.04
STATIONARY_SPEED_MPS = 0.08
MOVING_SPEED_MPS = 0.50

# Approximate pixel-to-meter conversion.
# Accurate real-world measurements require camera calibration.
REFERENCE_PERSON_HEIGHT_M = 1.70
REFERENCE_PIXEL_HEIGHT = 300.0

PIXELS_TO_METERS = (
    REFERENCE_PERSON_HEIGHT_M / (REFERENCE_PIXEL_HEIGHT * FRAME_SCALE)
)

# Height calibration
HEIGHT_CM = 185

# Possible-unconsciousness detection
GROUND_STILL_SECONDS = 5.0
STILL_THRESHOLD = 0.015

# Sudden movement detection
FALL_SPEED = 1.5
ALERT_SECONDS = 5.0

# Head movement detection
HEAD_MOVEMENT_SECONDS = 1.0
HEAD_MOVEMENT_RATIO = 0.15

# Landmark visibility threshold
VISIBILITY_THRESHOLD = 0.5


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


# Body connections only.
# No facial connections are drawn.
CONNECTIONS = [
    (11, 12),

    (11, 13), (13, 15),
    (12, 14), (14, 16),

    (15, 17), (15, 19), (15, 21),
    (16, 18), (16, 20), (16, 22),

    (11, 23), (12, 24), (23, 24),

    (23, 25), (25, 27),
    (24, 26), (26, 28),

    (27, 29), (27, 31),
    (28, 30), (28, 32)
]


# ============================================================
# PERSON TRACKING HISTORY
# ============================================================

person_history = defaultdict(
    lambda: deque(maxlen=300)
)

person_speed_history = defaultdict(
    lambda: deque(maxlen=SPEED_SMOOTHING_SAMPLES)
)

pose_track_history = defaultdict(
    lambda: {
        "prev_landmarks": None,
        "prev_time": None,

        "previous_posture": None,

        "ground_still_start": None,
        "fall_detected": False,

        "alert_until": 0.0,

        "calib_scale": None,

        "last_hip": None,
        "last_hip_time": None,

        "last_head_x": None,
        "last_head_y": None,
        "last_head_time": None,
    }
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def draw_text(
    frame,
    text,
    position,
    color=(0, 255, 0),
    scale=0.8
):
    """Draw readable text with a dark outline."""

    x, y = position
    font = cv2.FONT_HERSHEY_SIMPLEX

    thickness = max(2, int(scale * 2.5))

    # Dark outline for contrast.
    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        scale,
        (0, 0, 0),
        thickness + 3,
        cv2.LINE_AA
    )

    # Colored foreground text.
    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


def get_visible_landmarks(pose):
    return [
        i for i, landmark in enumerate(pose)
        if landmark.visibility > VISIBILITY_THRESHOLD
    ]


def get_pose_box(pose, frame_width, frame_height):
    visible_indices = get_visible_landmarks(pose)

    if len(visible_indices) < 10:
        return None

    points = []

    for i in visible_indices:
        landmark = pose[i]

        x = int(landmark.x * frame_width)
        y = int(landmark.y * frame_height)

        x = max(0, min(frame_width - 1, x))
        y = max(0, min(frame_height - 1, y))

        points.append((x, y))

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]

    return min(xs), min(ys), max(xs), max(ys)


def draw_pose(frame, pose):
    """Draw body skeleton connections without landmark dots."""

    frame_height, frame_width = frame.shape[:2]

    for first, second in CONNECTIONS:
        landmark_a = pose[first]
        landmark_b = pose[second]

        if (
            landmark_a.visibility > VISIBILITY_THRESHOLD
            and landmark_b.visibility > VISIBILITY_THRESHOLD
        ):
            point_a = (
                int(landmark_a.x * frame_width),
                int(landmark_a.y * frame_height)
            )

            point_b = (
                int(landmark_b.x * frame_width),
                int(landmark_b.y * frame_height)
            )

            cv2.line(
                frame,
                point_a,
                point_b,
                (255, 0, 0),
                3,
                cv2.LINE_AA
            )


# ============================================================
# POSTURE CLASSIFICATION
# ============================================================

def classify_posture(pose, pose_box):
    """
    Estimate whether a person is STANDING, SITTING, or LYING.

    This is a geometric heuristic, not a medical assessment.
    """

    if pose_box is None:
        return "UNKNOWN"

    x1, y1, x2, y2 = pose_box

    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)

    aspect_ratio = box_width / box_height

    def visible(index):
        return (
            pose[index].visibility > VISIBILITY_THRESHOLD
        )

    shoulders_visible = visible(11) and visible(12)
    hips_visible = visible(23) and visible(24)

    knees_visible = visible(25) and visible(26)
    ankles_visible = visible(27) and visible(28)

    # A wide, relatively horizontal body is likely lying down.
    if aspect_ratio > 1.15:
        return "LYING"

    if shoulders_visible and hips_visible:

        shoulder_y = (
            pose[11].y + pose[12].y
        ) / 2

        hip_y = (
            pose[23].y + pose[24].y
        ) / 2

        torso_vertical_fraction = (
            abs(hip_y - shoulder_y)
            / max(
                box_height / max(box_height, 1),
                1e-6
            )
        )

        if knees_visible and ankles_visible:

            left_hip_knee_gap = (
                pose[25].y - pose[23].y
            )

            right_hip_knee_gap = (
                pose[26].y - pose[24].y
            )

            average_hip_knee_gap = (
                left_hip_knee_gap + right_hip_knee_gap
            ) / 2

            # A compact pose may indicate sitting.
            if (
                average_hip_knee_gap < 0.12
                or torso_vertical_fraction < 0.38
            ):
                return "SITTING"

    return "STANDING"


# ============================================================
# LANDMARK MOVEMENT
# ============================================================

def normalized_landmark_movement(
    previous_landmarks,
    pose,
    frame_width,
    frame_height,
    normalization_size
):
    if previous_landmarks is None:
        return float("inf")

    displacements = []

    for i, landmark in enumerate(pose):
        previous = previous_landmarks[i]

        if (
            landmark.visibility > VISIBILITY_THRESHOLD
            and previous.visibility > VISIBILITY_THRESHOLD
        ):
            dx = (
                landmark.x - previous.x
            ) * frame_width

            dy = (
                landmark.y - previous.y
            ) * frame_height

            displacements.append(
                math.hypot(dx, dy)
            )

    if not displacements:
        return float("inf")

    average_displacement = (
        sum(displacements) / len(displacements)
    )

    return (
        average_displacement
        / max(normalization_size, 1)
    )


# ============================================================
# LOAD YOLO
# ============================================================

try:
    model = YOLO(YOLO_MODEL_PATH)

except Exception as error:
    print(f"ERROR: Could not load YOLO: {error}")
    sys.exit(1)


# ============================================================
# CHECK MEDIAPIPE MODEL
# ============================================================

if not os.path.isfile(POSE_MODEL_PATH):
    print(f"ERROR: Cannot find {POSE_MODEL_PATH}")
    print("Place the MediaPipe pose model beside main.py.")
    sys.exit(1)


# ============================================================
# OPEN CAMERA — macOS / iPHONE CONTINUITY CAMERA
# ============================================================

def open_camera(camera_index):
    """
    Open a camera using macOS AVFoundation.

    Apple's Continuity Camera can appear as a camera device
    when an iPhone is connected and available to the Mac.
    """

    print(f"Opening camera index {camera_index}...")

    cap = cv2.VideoCapture(
        camera_index,
        CAMERA_BACKEND
    )

    if cap.isOpened():
        success, test_frame = cap.read()

        if success and test_frame is not None:
            print(
                f"Camera {camera_index} opened successfully."
            )
            return cap

    cap.release()

    return None


camera = open_camera(CAMERA_INDEX)

if camera is None:
    print("\nERROR: Could not open the selected camera.")
    print("Check the following:")
    print("1. Your iPhone is nearby and unlocked.")
    print("2. Wi-Fi and Bluetooth are enabled.")
    print("3. Continuity Camera is enabled on your iPhone.")
    print("4. Your Mac can see your iPhone camera.")
    print("5. CAMERA_INDEX matches the iPhone camera.")
    print("\nTry CAMERA_INDEX = 1 or another available index.")

    sys.exit(1)


# ============================================================
# INITIALIZE MEDIAPIPE POSE LANDMARKER
# ============================================================

pose_landmarker = None

try:
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=POSE_MODEL_PATH
        ),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=MAX_POSES
    )

    pose_landmarker = (
        PoseLandmarker.create_from_options(options)
    )

except Exception as error:
    print(f"ERROR: Could not initialize MediaPipe: {error}")
    camera.release()
    sys.exit(1)


last_timestamp_ms = -1


# ============================================================
# MAIN LOOP
# ============================================================

try:
    while True:

        # ----------------------------------------------------
        # READ CAMERA
        # ----------------------------------------------------

        success, frame = camera.read()

        if not success:
            print("ERROR: Could not read camera frame.")
            break

        # Mirror the camera image horizontally
        if MIRROR_CAMERA:
            frame = cv2.flip(frame, 1)

        # Scale frame (e.g. 0.5x)
        if FRAME_SCALE != 1.0:
            frame = cv2.resize(
                frame,
                (0, 0),
                fx=FRAME_SCALE,
                fy=FRAME_SCALE,
                interpolation=cv2.INTER_LINEAR
            )

        current_time = time.monotonic()

        frame_height, frame_width = frame.shape[:2]


        # ----------------------------------------------------
        # YOLO PERSON DETECTION AND TRACKING
        # ----------------------------------------------------

        results = model.track(
            frame,
            persist=True,
            verbose=False
        )

        result = results[0]

        yolo_people = []

        if (
            result.boxes is not None
            and result.boxes.id is not None
        ):
            boxes = result.boxes.xyxy.cpu().numpy()

            track_ids = (
                result.boxes.id.int().cpu().tolist()
            )

            classes = (
                result.boxes.cls.int().cpu().tolist()
            )

            confidences = (
                result.boxes.conf.cpu().tolist()
            )

            for box, track_id, class_id, confidence in zip(
                boxes,
                track_ids,
                classes,
                confidences
            ):
                if model.names[class_id] != "person":
                    continue

                x1, y1, x2, y2 = box

                yolo_people.append({
                    "id": track_id,
                    "box": (x1, y1, x2, y2),
                    "confidence": confidence,
                    "center": (
                        (x1 + x2) / 2,
                        (y1 + y2) / 2
                    )
                })


        # ----------------------------------------------------
        # MEDIAPIPE POSE ESTIMATION
        # ----------------------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp_ms = int(time.monotonic() * 1000)

        timestamp_ms = max(
            timestamp_ms,
            last_timestamp_ms + 1
        )

        last_timestamp_ms = timestamp_ms

        pose_result = pose_landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        poses = pose_result.pose_landmarks

        pose_boxes = []
        pose_centers = []

        for pose in poses:
            pose_box = get_pose_box(
                pose,
                frame_width,
                frame_height
            )

            pose_boxes.append(pose_box)

            if pose_box is None:
                pose_centers.append(None)
            else:
                px1, py1, px2, py2 = pose_box

                pose_centers.append((
                    (px1 + px2) / 2,
                    (py1 + py2) / 2
                ))


        # ----------------------------------------------------
        # MATCH POSES TO YOLO TRACKING IDS
        # ----------------------------------------------------

        pose_to_track = {}
        track_to_pose = {}

        candidate_matches = []

        for pose_index, pose_center in enumerate(pose_centers):
            if pose_center is None:
                continue

            for person in yolo_people:
                person_center = person["center"]

                distance = math.hypot(
                    pose_center[0] - person_center[0],
                    pose_center[1] - person_center[1]
                )

                candidate_matches.append((
                    distance,
                    pose_index,
                    person["id"]
                ))

        candidate_matches.sort(
            key=lambda item: item[0]
        )

        used_pose_indices = set()
        used_track_ids = set()

        for distance, pose_index, track_id in candidate_matches:

            if pose_index in used_pose_indices:
                continue

            if track_id in used_track_ids:
                continue

            pose_to_track[pose_index] = track_id
            track_to_pose[track_id] = pose_index

            used_pose_indices.add(pose_index)
            used_track_ids.add(track_id)


        # Draw body skeletons without facial lines or dots.
        for pose in poses:
            draw_pose(frame, pose)


        # ----------------------------------------------------
        # PROCESS EACH PERSON
        # ----------------------------------------------------

        calibration_candidates = {}

        for person in yolo_people:

            track_id = person["id"]
            confidence = person["confidence"]

            x1, y1, x2, y2 = person["box"]

            width = max(0, x2 - x1)
            height = max(0, y2 - y1)

            x1i, y1i, x2i, y2i = map(
                int,
                (x1, y1, x2, y2)
            )

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)


            # =================================================
            # MOVEMENT SPEED
            # Calculated internally; not displayed on screen.
            # =================================================

            width_m = width * PIXELS_TO_METERS
            height_m = height * PIXELS_TO_METERS

            person_history[track_id].append((
                current_time,
                center_x,
                center_y
            ))

            history = person_history[track_id]

            while (
                history
                and current_time - history[0][0]
                > HISTORY_SECONDS
            ):
                history.popleft()

            recent_history = [
                point for point in history
                if current_time - point[0]
                <= MOVEMENT_WINDOW_SECONDS
            ]

            total_distance_meters = 0.0

            for i in range(1, len(recent_history)):

                previous = recent_history[i - 1]
                current = recent_history[i]

                distance_pixels = math.hypot(
                    current[1] - previous[1],
                    current[2] - previous[2]
                )

                distance_meters = (
                    distance_pixels * PIXELS_TO_METERS
                )

                if distance_meters >= MIN_MOVEMENT_METERS:
                    total_distance_meters += distance_meters

            speed_mps = 0.0

            if len(recent_history) >= 2:

                elapsed_time = (
                    recent_history[-1][0]
                    - recent_history[0][0]
                )

                if elapsed_time > 0:

                    raw_speed_mps = (
                        total_distance_meters
                        / elapsed_time
                    )

                    person_speed_history[track_id].append(
                        raw_speed_mps
                    )

                    speed_history = (
                        person_speed_history[track_id]
                    )

                    speed_mps = (
                        sum(speed_history)
                        / len(speed_history)
                    )

            if speed_mps < STATIONARY_SPEED_MPS:
                movement_status = "STATIONARY"

            elif speed_mps < MOVING_SPEED_MPS:
                movement_status = "MOVING"

            else:
                movement_status = "FAST"


            # =================================================
            # MOVEMENT DIRECTION
            # Calculated internally; not displayed on screen.
            # =================================================

            direction = "STATIONARY"

            if len(recent_history) >= 2:

                first_x = recent_history[0][1]
                first_y = recent_history[0][2]

                dx = center_x - first_x
                dy = center_y - first_y

                if math.hypot(dx, dy) > 30:

                    if abs(dx) > abs(dy):
                        direction = (
                            "RIGHT" if dx > 0 else "LEFT"
                        )

                    else:
                        direction = (
                            "DOWN" if dy > 0 else "UP"
                        )


            # =================================================
            # DRAW PERSON BOX AND LABEL
            # =================================================

            cv2.rectangle(
                frame,
                (x1i, y1i),
                (x2i, y2i),
                (0, 255, 0),
                2
            )

            cv2.circle(
                frame,
                (center_x, center_y),
                5,
                (0, 0, 255),
                -1
            )

            draw_text(
                frame,
                f"Person #{track_id}",
                (x1i, max(35, y1i - 15)),
                scale=0.9
            )


            # =================================================
            # POSTURE AND POSSIBLE-UNCONSCIOUSNESS DETECTION
            # =================================================

            state = pose_track_history[track_id]

            pose_index = track_to_pose.get(track_id)

            label_y = min(
                frame_height - 20,
                y2i + 45
            )

            if pose_index is not None:

                pose = poses[pose_index]
                pose_box = pose_boxes[pose_index]

                if pose_box is not None:

                    px1, py1, px2, py2 = pose_box

                    pose_box_width = max(
                        1,
                        px2 - px1
                    )

                    pose_box_height = max(
                        1,
                        py2 - py1
                    )

                    visible_indices = get_visible_landmarks(
                        pose
                    )

                    if len(visible_indices) >= 10:

                        # -------------------------------------
                        # 1. CLASSIFY POSTURE
                        # -------------------------------------

                        posture = classify_posture(
                            pose,
                            pose_box
                        )

                        draw_text(
                            frame,
                            f"Posture: {posture}",
                            (x1i, label_y),
                            color=(255, 150, 0),
                            scale=0.8
                        )

                        label_y = min(
                            frame_height - 10,
                            label_y + 30
                        )


                        # -------------------------------------
                        # 2. MEASURE BODY MOVEMENT
                        # -------------------------------------

                        previous_landmarks = (
                            state["prev_landmarks"]
                        )

                        previous_posture = (
                            state["previous_posture"]
                        )

                        movement_norm = (
                            normalized_landmark_movement(
                                previous_landmarks,
                                pose,
                                frame_width,
                                frame_height,
                                max(
                                    pose_box_width,
                                    pose_box_height
                                )
                            )
                        )


                        # -------------------------------------
                        # 3. DETECT SUDDEN HEAD MOVEMENT
                        # -------------------------------------

                        # The nose is used as a body landmark.
                        # No facial recognition or identity matching.

                        nose = pose[0]

                        if (
                            nose.visibility
                            > VISIBILITY_THRESHOLD
                        ):

                            head_x = nose.x * frame_width
                            head_y = nose.y * frame_height

                            last_head_x = state["last_head_x"]
                            last_head_y = state["last_head_y"]
                            last_head_time = state["last_head_time"]

                            if (
                                last_head_x is not None
                                and last_head_y is not None
                                and last_head_time is not None
                                and previous_posture in (
                                    "SITTING",
                                    "STANDING"
                                )
                            ):

                                head_dt = (
                                    current_time
                                    - last_head_time
                                )

                                head_shift = math.hypot(
                                    head_x - last_head_x,
                                    head_y - last_head_y
                                )

                                if (
                                    0 < head_dt
                                    <= HEAD_MOVEMENT_SECONDS
                                    and head_shift
                                    > pose_box_height
                                    * HEAD_MOVEMENT_RATIO
                                ):

                                    state["alert_until"] = (
                                        current_time
                                        + ALERT_SECONDS
                                    )

                            state["last_head_x"] = head_x
                            state["last_head_y"] = head_y
                            state["last_head_time"] = current_time

                        else:

                            state["last_head_x"] = None
                            state["last_head_y"] = None
                            state["last_head_time"] = None


                        # -------------------------------------
                        # 4. DETECT RAPID DOWNWARD MOVEMENT
                        # -------------------------------------

                        hips_visible = (
                            pose[23].visibility
                            > VISIBILITY_THRESHOLD
                            and pose[24].visibility
                            > VISIBILITY_THRESHOLD
                        )

                        if hips_visible:

                            hip_y = (
                                pose[23].y * frame_height
                                + pose[24].y * frame_height
                            ) / 2

                            previous_hip = state["last_hip"]

                            previous_hip_time = (
                                state["last_hip_time"]
                            )

                            if (
                                previous_hip is not None
                                and previous_hip_time is not None
                            ):

                                dt = (
                                    current_time
                                    - previous_hip_time
                                )

                                if (
                                    dt > 0
                                    and pose_box_height > 0
                                ):

                                    normalized_vertical_speed = (
                                        (hip_y - previous_hip)
                                        / dt
                                        / pose_box_height
                                    )

                                    if (
                                        normalized_vertical_speed
                                        > FALL_SPEED
                                    ):

                                        # A rapid drop is a suspected
                                        # fall, not proof of unconsciousness.
                                        state["fall_detected"] = True

                            state["last_hip"] = hip_y
                            state["last_hip_time"] = current_time

                        else:

                            state["last_hip"] = None
                            state["last_hip_time"] = None


                        # -------------------------------------
                        # 5. PERSON LYING ON THE GROUND
                        # -------------------------------------

                        if posture == "LYING":

                            if state["ground_still_start"] is None:
                                state["ground_still_start"] = (
                                    current_time
                                )

                            still_duration = (
                                current_time
                                - state["ground_still_start"]
                            )

                            # Alert after the person has remained
                            # lying with very little movement for 5 seconds.
                            if (
                                still_duration
                                >= GROUND_STILL_SECONDS
                                and movement_norm
                                < STILL_THRESHOLD
                            ):

                                state["alert_until"] = max(
                                    state["alert_until"],
                                    current_time + 0.5
                                )

                        else:

                            state["ground_still_start"] = None
                            state["fall_detected"] = False


                        # -------------------------------------
                        # 6. DISPLAY ALERT
                        # -------------------------------------

                        if current_time < state["alert_until"]:

                            draw_text(
                                frame,
                                "POSSIBLE UNCONSCIOUSNESS",
                                (max(10, x1i), max(45, label_y)),
                                color=(0, 0, 255),
                                scale=1.0
                            )

                            label_y = min(
                                frame_height - 10,
                                label_y + 30
                            )


                        # -------------------------------------
                        # SAVE CURRENT POSE HISTORY
                        # -------------------------------------

                        state["prev_landmarks"] = pose
                        state["prev_time"] = current_time
                        state["previous_posture"] = posture

                        calibration_candidates[track_id] = {
                            "posture": posture,
                            "height": pose_box_height
                        }

                    else:

                        # Reset pose history when landmarks are unreliable.
                        state["prev_landmarks"] = None
                        state["prev_time"] = None
                        state["previous_posture"] = None
                        state["ground_still_start"] = None
                        state["last_hip"] = None
                        state["last_hip_time"] = None
                        state["last_head_x"] = None
                        state["last_head_y"] = None
                        state["last_head_time"] = None

                else:

                    state["prev_landmarks"] = None
                    state["prev_time"] = None
                    state["previous_posture"] = None
                    state["ground_still_start"] = None
                    state["last_hip"] = None
                    state["last_hip_time"] = None
                    state["last_head_x"] = None
                    state["last_head_y"] = None
                    state["last_head_time"] = None


            # -----------------------------------------------
            # TERMINAL OUTPUT
            # -----------------------------------------------

            print(
                f"ID={track_id} "
                f"Conf={confidence:.2f} "
                f"W={width_m:.2f}m "
                f"H={height_m:.2f}m "
                f"Speed={speed_mps:.2f}m/s "
                f"{movement_status}"
            )



        # ====================================================
        # DISPLAY FRAME
        # ====================================================

        cv2.imshow(
            "YOLO + MediaPipe Emergency Detection",
            frame
        )


        # ====================================================
        # KEYBOARD CONTROLS
        # ====================================================

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("m"), ord("M")):
            MIRROR_CAMERA = not MIRROR_CAMERA
            print(f"Camera mirror toggled: {'ON' if MIRROR_CAMERA else 'OFF'}")

        if key in (ord("c"), ord("C")):

            calibrated_count = 0

            for track_id, candidate in (
                calibration_candidates.items()
            ):

                if (
                    candidate["posture"] == "STANDING"
                    and candidate["height"] > 0
                ):

                    state = pose_track_history[track_id]

                    state["calib_scale"] = (
                        HEIGHT_CM / candidate["height"]
                    )

                    calibrated_count += 1

            if calibrated_count:
                print(
                    "Height calibration completed for "
                    f"{calibrated_count} person(s)."
                )

            else:
                print(
                    "Calibration not applied. "
                    "Make sure a standing person is detected."
                )

        if key in (ord("q"), 27):
            break


# ============================================================
# CLEANUP
# ============================================================

finally:

    if pose_landmarker is not None:
        pose_landmarker.close()

    camera.release()
    cv2.destroyAllWindows()