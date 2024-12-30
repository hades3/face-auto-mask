import numpy as np
import mediapipe as mp
import cv2
mp_face_detection = mp.solutions.face_detection # 얼굴 탐지
mp_drawing = mp.solutions.drawing_utils # 얼굴에 특징 그리기

cap = cv2.VideoCapture('../resources/talking.mp4')  # 비디오 파일을 열어서 객체로 저장
with mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.9) as face_detection: # 대상과의 거리에 따른 모델 종류 선택, 임계치
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = face_detection.process(image) # 검출 결과 반환

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.detections:
            for detection in results.detections:    # 우측 눈, 좌측 눈, 코, 입 중앙, 우측 귀, 좌측 귀
                mp_drawing.draw_detection(image, detection)

                keypoints = detection.location_data.relative_keypoints
                right_eye = keypoints[0]
                left_eye = keypoints[1]

                h, w, _ = image.shape   # height, width, channel

                right_eye = (int(right_eye.x * w), int(right_eye.y * h))
                left_eye = (int(left_eye.x * w), int(left_eye.y * h))

                cv2.circle(image, right_eye, 30, (255, 0, 0), 3, cv2.LINE_AA)
                cv2.circle(image, left_eye, 30, (0, 255, 0), 3, cv2.LINE_AA)

        cv2.imshow('Face Detection', cv2.resize(image, None, fx=0.5, fy=0.5))

        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()