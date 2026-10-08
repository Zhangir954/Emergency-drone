import time
import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = r"C:\Users\User\pose_landmarker_lite.task"
HEIGHT_CM = 185        # ваш реальный рост для калибровки
STILL_SECONDS = 3      # сколько секунд неподвижности считать тревогой
STILL_THRESHOLD = 0.01 # порог движения (меньше = неподвижен)

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (27, 29), (27, 31), (28, 30), (28, 32),
]

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=1,
)

cap = cv2.VideoCapture(0)
timestamp_ms = 0
prev_pts = None
still_since = None
cm_per_px = None
prev_hip_y = None
prev_time = None
fall_until = 0
FALL_SPEED = 1.5
with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms += 33
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        box_h = 0
        lying = False

        if result.pose_landmarks:
            h, w, _ = frame.shape
            lms = result.pose_landmarks[0]
            pts = [(l.x * w, l.y * h) for l in lms]
            good = [p for p, l in zip(pts, lms) if l.visibility > 0.5]

            for a, b in CONNECTIONS:
                cv2.line(frame, (int(pts[a][0]), int(pts[a][1])),
                         (int(pts[b][0]), int(pts[b][1])), (0, 255, 0), 2)
            for p in pts:
                cv2.circle(frame, (int(p[0]), int(p[1])), 4, (0, 0, 255), -1)

            if len(good) >= 10:
                xs = [p[0] for p in good]
                ys = [p[1] for p in good]
                box_w = max(xs) - min(xs)
                box_h = max(ys) - min(ys)
                lying = box_w > box_h * 1.1
                hip_y = (pts[23][1] + pts[24][1]) / 2
                now = time.time()
                if prev_hip_y is not None and box_h > 0:
                    dt = now - prev_time
                    speed = (hip_y - prev_hip_y) / dt / max(box_h, 1)
                    if speed > FALL_SPEED:
                        fall_until = now + 3
                prev_hip_y, prev_time = hip_y, now

                if now < fall_until:
                    cv2.putText(frame, "FALL DETECTED", (10, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                if prev_pts is not None:
                    d = sum(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
                            for a, b in zip(pts, prev_pts)) / len(pts)
                    movement = d / max(box_w, box_h, 1)
                else:
                    movement = 1
                prev_pts = pts

                if lying and movement < STILL_THRESHOLD:
                    if still_since is None:
                        still_since = time.time()
                    still_for = time.time() - still_since
                else:
                    still_since = None
                    still_for = 0

                if cm_per_px:
                    height_text = f"Height: ~{box_h * cm_per_px:.0f} cm"
                else:
                    height_text = f"Height: {box_h:.0f} px (press c)"

                cv2.putText(frame, height_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                cv2.putText(frame, "LYING" if lying else "STANDING", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                if still_for >= STILL_SECONDS:
                    cv2.putText(frame, "POSSIBLE UNCONSCIOUS", (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            prev_pts = None
            still_since = None

        cv2.imshow("Pose Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("c") and box_h > 0 and not lying:
            cm_per_px = HEIGHT_CM / box_h
        if key == ord("q") or key == 27:
            break

cap.release()   
cv2.destroyAllWindows()