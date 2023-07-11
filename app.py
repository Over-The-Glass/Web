import dlib
import cv2
import numpy as np
import os
from collections import deque
from flask import Flask, render_template, Response, jsonify, request
import asyncio

app = Flask(__name__)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

client_speech = ""

class LipMovement:
    def __init__(self, name):
        self.name = name
        self.width_diffs = deque(maxlen=10)
        self.height_diffs = deque(maxlen=10)
        self.prev_height = 0
        self.prev_width = 0

    def check_movement(self, width, height):
        self.height_diffs.append(abs(self.prev_height - height))
        self.width_diffs.append(abs(self.prev_width - width))
        self.width_numbers = list(self.width_diffs)
        self.height_numbers = list(self.height_diffs)
        self.width_average = sum(self.width_numbers) / len(self.width_numbers)
        self.height_average = sum(self.height_numbers) / len(self.height_numbers)
        self.prev_height = height
        self.prev_width = width
        return round(self.width_average, 3), round(self.height_average, 3)

known_faces = []
known_names = []

for file in os.listdir("faces"):
    image = cv2.imread(os.path.join("faces", file))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) > 0:
        landmarks = predictor(gray, faces[0])
        face_encoding = np.array(facerec.compute_face_descriptor(image, landmarks))
        known_faces.append(face_encoding)
        known_names.append(file.split(".")[0])
    else:
        print('No face found in', file)

print(f"Loaded face data of {known_names}")
number_of_known_people = len(known_names)

movements = [LipMovement(known_names[i]) for i in range(len(known_names))]

check_frame = True
latest_speaker_position = ()
difference = [0 for _ in range(len(known_names))]
name = "?"

def process_frame(data):
    global latest_speaker_position
    global name

    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    frame = cv2.flip(image, 0)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    has_speaker = False
    landmarks = None

    if not faces:  # 얼굴이 감지되지 않을 경우
        name = "Unknown"  # Unknown으로 표시


    for face in faces:
        landmarks = predictor(gray, face)

        face_encoding = np.array(facerec.compute_face_descriptor(frame, landmarks))

        face_distances = []
        for known_face in known_faces:
            face_distances.append(np.linalg.norm(known_face - face_encoding))

        min_distance = min(face_distances)
        min_distance_index = np.argmin(face_distances)
        name = known_names[min_distance_index]
        match_rate = 1 / (1 + min_distance)

        if match_rate > 0.5:
            landmarks_array = np.array([[p.x, p.y] for p in landmarks.parts()])
            lip_height = np.linalg.norm(landmarks_array[62] - landmarks_array[66])
            lip_width = np.linalg.norm(landmarks_array[48] - landmarks_array[54])

            eye_width = np.linalg.norm(landmarks_array[37] - landmarks_array[44])

            d = eye_width * 0.06

            width_average, height_average = movements[min_distance_index].check_movement(
                lip_width / eye_width * 100,
                lip_height / eye_width * 100)
            # text = f"w: {width_average}, h: {height_average}"

            if height_average > 3 or width_average > 3:
                # cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 1)
                difference[min_distance_index] = height_average + width_average
                has_speaker = True

            if np.argmax(difference) == min_distance_index:
                latest_speaker_position = (face.left(), face.bottom() + 60)

            # if latest_speaker_position:
            #     # draw_text(frame, client_speech, pos=latest_speaker_position)

            # cv2.putText(frame, f"{name} ({match_rate:.2%})", (face.left(), face.bottom() + 20),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        else:  # 매치율이 0.5보다 작을 경우
            name = "Unknown"  # Unknown으로 표시
            # cv2.putText(frame, f"{name}", (face.left(), face.bottom() + 20),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    if not has_speaker and client_speech:
        if not faces:  # 얼굴이 감지되지 않을 경우
            name = "Unknown"
        # draw_text(frame, client_speech, pos=(frame.shape[1] // 2, frame.shape[0] - 30))

    return

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/conversation-mode')
def conversation_mode():
    return render_template('conversation-mode.html')

@app.route('/menu')
def menu():
    return render_template('menu.html')

@app.route('/speaker_info')
def speaker_info():
    return jsonify({'speaker': name})

@app.route('/process_speech', methods=['POST'])
def process_speech():
    global client_speech
    client_speech = request.json['speech']
    return 'OK'

@app.route('/camera', methods=['POST'])
def camera():
    if request.method == 'POST':
        # 요청에서 카메라 프레임 데이터를 가져옵니다.
        frame_data = np.frombuffer(request.data, dtype=np.uint8)

        # 프레임 데이터를 처리하기 전에 로그 문장을 추가합니다.
        print('Received camera frame:', len(frame_data), 'bytes')

        # 프레임 데이터를 처리하고 처리된 프레임을 JPEG 형식으로 얻습니다.
        process_frame(frame_data)
        print(name)

        # 응답으로는 프레임 데이터가 아닌 성공 상태를 반환합니다.
        return 'Success'
    else:
        # 다른 메서드의 경우 오류 응답을 반환합니다.
        return '허용되지 않는 메서드입니다.', 405

if __name__ == '__main__':
    app.run(threaded=True)