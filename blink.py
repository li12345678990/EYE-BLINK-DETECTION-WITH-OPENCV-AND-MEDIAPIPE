import mediapipe as mp
import cv2
import numpy as np
from collections import deque

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 眼睛关键点索引
LEFT_EYE_POINTS = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_POINTS = [33, 160, 158, 133, 153, 144]


EAR_THRESHOLD = 0.2  # 眨眼阈值
EAR_CONSEC_FRAMES = 5  # 连续帧数阈值
BLINK_TOTAL = 0  # 眨眼总数
EAR_BUFFER = deque(maxlen=5)  # 平滑缓冲区


def calculate_ear(eye_points, landmarks, image_width, image_height):

    p1 = landmarks[eye_points[0]]
    p2 = landmarks[eye_points[1]]
    p3 = landmarks[eye_points[2]]
    p4 = landmarks[eye_points[3]]
    p5 = landmarks[eye_points[4]]
    p6 = landmarks[eye_points[5]]

    p1 = (int(p1.x * image_width), int(p1.y * image_height))
    p2 = (int(p2.x * image_width), int(p2.y * image_height))
    p3 = (int(p3.x * image_width), int(p3.y * image_height))
    p4 = (int(p4.x * image_width), int(p4.y * image_height))
    p5 = (int(p5.x * image_width), int(p5.y * image_height))
    p6 = (int(p6.x * image_width), int(p6.y * image_height))

    # 计算距离
    A = np.linalg.norm(np.array(p2) - np.array(p6))
    B = np.linalg.norm(np.array(p3) - np.array(p5))
    C = np.linalg.norm(np.array(p1) - np.array(p4))
    ear = (A + B) / (2.0 * C)

    return ear, [p1, p2, p3, p4, p5, p6]



video_path = "test.mp4"
output_path = "output.mp4"

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

BLINK_COUNTER = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        frame_height, frame_width = frame.shape[:2]

        left_ear, left_eye_points = calculate_ear(
            LEFT_EYE_POINTS, face_landmarks.landmark, frame_width, frame_height
        )
        right_ear, right_eye_points = calculate_ear(
            RIGHT_EYE_POINTS, face_landmarks.landmark, frame_width, frame_height
        )

        ear = (left_ear + right_ear) / 2.0
        EAR_BUFFER.append(ear)
        smoothed_ear = sum(EAR_BUFFER) / len(EAR_BUFFER) if EAR_BUFFER else ear

        if smoothed_ear < EAR_THRESHOLD:
            BLINK_COUNTER += 1
        else:
            if BLINK_COUNTER >= EAR_CONSEC_FRAMES:
                BLINK_TOTAL += 1
            BLINK_COUNTER = 0

        cv2.polylines(
            frame,
            [np.array(left_eye_points)],
            isClosed=True,
            color=(0, 255, 0),
            thickness=2
        )
        cv2.polylines(
            frame,
            [np.array(right_eye_points)],
            isClosed=True,
            color=(0, 255, 0),
            thickness=2
        )

        cv2.putText(
            frame, f"EAR: {smoothed_ear:.2f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )
        cv2.putText(
            frame, f"Blinks: {BLINK_TOTAL}", (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )
        state = "Blinking" if smoothed_ear < EAR_THRESHOLD else "Open"
        cv2.putText(
            frame, f"State: {state}", (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )


    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()