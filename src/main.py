import numpy as np
import mediapipe as mp
import cv2

mp_face_detection = mp.solutions.face_detection  # 얼굴 탐지
mp_drawing = mp.solutions.drawing_utils  # 얼굴에 특징 그리기


def overlay(src_image, x, y, w, h, overlay_image):
    alpha = overlay_image[:, :, 3]
    mask_image = alpha / 255  # 0~1 사이의 값으로 변환(0은 투명, 1은 불투명)

    for c in range(0, 3):
        # + 뒤의 식을 쓰지 않으면, 채워지지 않음
        src_image[y-h:y+h, x-w:x+w, c] = (overlay_image[:, :, c] * mask_image) + (src_image[y-h:y+h, x-w:x+w, c] * (1 - mask_image))


cap = cv2.VideoCapture('../resources/talking.mp4')  # 비디오 파일을 열어서 객체로 저장

image_left_eye = cv2.imread('../resources/cat_left.jpg', cv2.IMREAD_UNCHANGED)
image_right_eye = cv2.imread('../resources/cat_right.jpg', cv2.IMREAD_UNCHANGED)
image_nose = cv2.imread('../resources/cat_mid.jpg', cv2.IMREAD_UNCHANGED)

with mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.9) as face_detection:  # 대상과의 거리에 따른 모델 종류 선택, 임계치
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = face_detection.process(image)  # 검출 결과 반환

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.detections:
            for detection in results.detections:  # 우측 눈, 좌측 눈, 코, 입 중앙, 우측 귀, 좌측 귀
                mp_drawing.draw_detection(image, detection)

                keypoints = detection.location_data.relative_keypoints
                right_eye = keypoints[0]
                left_eye = keypoints[1]
                nose = keypoints[2]

                h, w, _ = image.shape  # height, width, channel

                right_eye = (int(right_eye.x * w) - 50, int(right_eye.y * h) - 100)
                left_eye = (int(left_eye.x * w) + 50, int(left_eye.y * h) - 100)
                nose = (int(nose.x * w), int(nose.y * h))

                overlay(image, *right_eye, 90, 90, image_right_eye)
                overlay(image, *left_eye, 90, 90, image_left_eye)
                overlay(image, *nose, 100, 100, image_nose)

        cv2.imshow('Face Detection', cv2.resize(image, None, fx=0.5, fy=0.5))

        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
