# 코드 리팩토링, 발화자 파악 개선(발화자 기록 객체 추가), 기본 자막 위치(중앙 하단) 설정
import dlib
import cv2
import numpy as np
import os
from collections import deque

# 파일 경로와 이름, 실행 환경 설정(colab일 경우 is_real_tiem, input_video_name, output_video_name 바꾸기)
is_real_time = True
shape_predictor_path = "shape_predictor_68_face_landmarks.dat"
face_recognition_model_path = "dlib_face_recognition_resnet_model_v1.dat"
face_image_file_path = "faces"
input_path = ""
input_video_name = ".mp4"
output_video_name = ".avi"

if is_real_time == False:
    colab_source_path = "/content/drive/MyDrive/Colab_Sources/"
    face_image_file_path = colab_source_path + face_image_file_path
    shape_predictor_path = colab_source_path + shape_predictor_path
    face_recognition_model_path = colab_source_path + face_recognition_model_path
    input_path = colab_source_path + input_video_name
    output_path = colab_source_path + output_video_name

# 지속적인 움직임 확인
class LipMovement:
    # 클래스 생성자 정의
    def __init__(self, name):
        self.name = name
        # deque 최대 길이를 3개로 유지
        self.width_diffs = deque(maxlen=3)
        self.height_diffs = deque(maxlen=3)
        # 이전 프레임 속 입의 세로 가로 길이
        self.prev_height = 0
        self.prev_width = 0

    # 입 길이 변화의 평균값 반환
    def check_movement(self, width, height):
        self.height_diffs.append(abs(self.prev_height - height))
        self.width_diffs.append(abs(self.prev_width - width))
        self.width_average = sum(self.width_diffs) / len(self.width_diffs)
        self.height_average = sum(self.height_diffs) / len(self.height_diffs)
        self.prev_height = height
        self.prev_width = width
        return round(self.width_average, 3), round(self.height_average, 3)

# 글씨 크기, 색상 설정하여 자막 출력하는 함수
def draw_text(img, text,
          font=cv2.FONT_HERSHEY_SIMPLEX,
          pos=(0, 0),
          font_scale=1,
          font_thickness=2,
          text_color=(255, 255, 255),
          bg_color=(0, 0, 0)
          ):
    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), bg_color, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)
    return


# Load face detection and landmark detection models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)

# Load face recognition model
facerec = dlib.face_recognition_model_v1(face_recognition_model_path)

# Load the known faces and embeddings
known_faces = []
known_names = []

for file in os.listdir(face_image_file_path):
    image = cv2.imread(os.path.join(face_image_file_path, file))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) > 0:
        landmarks = predictor(gray, faces[0]) # 이미지에 한 사람이 있다고 가정하고 첫번째 얼굴만 사용
        face_encoding = np.array(facerec.compute_face_descriptor(image, landmarks))
        known_faces.append(face_encoding)
        known_names.append(file.split(".")[0])
    else:
        print('No face found in', file)

print(f"Loaded face data of {known_names}")
number_of_known_people = len(known_names)

# 움직임 확인 객체 생성
movements = [LipMovement(known_names[i]) for i in range(number_of_known_people)]

check_frame = True
latest_speaker_position = ()
difference = [0 for i in range(number_of_known_people)]

# Capture video
if is_real_time:
    video_capture = cv2.VideoCapture(0)
else:
    video_capture = cv2.VideoCapture(input_path)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), 25, (int(video_capture.get(3)), int(video_capture.get(4))))

while video_capture.isOpened():
    # Read a frame from the video capture
    ret, frame = video_capture.read()

    if not ret:
        break

    if check_frame:
        difference = [0 for i in range(number_of_known_people)]

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = detector(gray)

        # Iterate through the faces
        for face in faces:
            name = "?"
            landmarks = predictor(gray, face)

            face_encoding = np.array(facerec.compute_face_descriptor(frame, landmarks))

            # Find the closest matching known face
            face_distances = []
            for known_face in known_faces:
                face_distances.append(np.linalg.norm(known_face - face_encoding))

            min_distance = min(face_distances)
            min_distance_index = np.argmin(face_distances)  # np.argmin -> 최소값 색인 위치
            name = known_names[min_distance_index]
            match_rate = 1 / (1 + min_distance)

            # 얼굴 인식 매칭률이 높을 경우
            if match_rate > 0.5:
                landmarks_array = np.array([[p.x, p.y] for p in landmarks.parts()])

                # 입술 사이 거리
                lip_height = np.linalg.norm(landmarks_array[62] - landmarks_array[66])
                lip_width = np.linalg.norm(landmarks_array[48] - landmarks_array[54])

                # 눈 사이 거리
                eye_width = np.linalg.norm(landmarks_array[37] - landmarks_array[44])

                # 얼굴 크기에 따른 입 벌어짐 판단 기준
                d = eye_width * 0.06

                # 3프레임동안 입 길이 변화 평균값 출력
                width_average, height_average = movements[min_distance_index].check_movement(
                    lip_width / eye_width * 100, lip_height / eye_width * 100)  # 눈 사이 거리에 따른 비율 적용
                text = "w: " + str(width_average) + ", h: " + str(height_average)

                # 가로 또는 세로 길이 변화 평균값이 일정 정도 이상이 될 경우 말하고 있다고 판단
                if (height_average > 2 or width_average > 2):
                    # 일정 정도 이상의 입 움직임이 감지될 경우 얼굴에 사각형 그리기
                    cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 1)
                    difference[min_distance_index] = height_average + width_average

                if np.argmax(difference) == min_distance_index:
                    latest_speaker_position = (face.left(), face.bottom() + 60)

                # Draw text on the frame
                cv2.putText(frame, text, (face.left(), face.bottom() + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Draw text on the frame
            cv2.putText(frame, f"{name} ({match_rate:.2%})", (face.left(), face.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    else:
        check_frame = not check_frame

    if latest_speaker_position:
        # cv2.putText(frame, "subtitle location", latest_speaker_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        draw_text(frame, "subtitle location", pos=latest_speaker_position)

    # Display the resulting frame
    if is_real_time:
        cv2.imshow('Video', frame)
    else:
        out.write(frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
if not is_real_time:
    out.release()
else:
    cv2.destroyAllWindows()
