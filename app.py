import random
import dlib
import cv2
import numpy as np
import os
import pymysql
from collections import deque
from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, join_room, leave_room, emit

app = Flask(__name__)
socketio = SocketIO(app)

db = pymysql.connect(host='localhost', user='root', password='0000', db='db')

with db.cursor() as cursor:
    try:
        # 예시로 데이터베이스에 있는 테이블 목록을 가져옵니다.
        cursor.execute('SHOW TABLES')
        tables = cursor.fetchall()
        table_names = [table[0] for table in tables]
        print(f'Database connection successful! Tables: {", ".join(table_names)}')
    except Exception as e:
        print(f'Database connection failed: {e}')


# 방과 사용자를 매핑할 딕셔너리를 생성합니다.
rooms = {}

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

client_speech = ""
is_processing = False

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
    global is_processing 
    global name

    is_processing = True

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

            if height_average > 3 or width_average > 3:
                difference[min_distance_index] = height_average + width_average
                has_speaker = True

            if np.argmax(difference) == min_distance_index:
                latest_speaker_position = (face.left(), face.bottom() + 60)
        
        else:  # 매치율이 0.5보다 작을 경우
            name = "Unknown"  # Unknown으로 표시

    if not has_speaker and client_speech:
        if not faces:  # 얼굴이 감지되지 않을 경우
            name = "Unknown"

    is_processing = False

    return

@app.route('/')
def main():
    return render_template('main.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/signup', methods=['GET'])
def signup():
    return render_template('signup.html')


@app.route('/chatroom')
def chatroom():
    return render_template('chatroom.html')

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
        global is_processing 

        # 프레임 데이터를 처리하고 처리된 프레임을 JPEG 형식으로 얻습니다.
        if is_processing == True :
            print("already processing")
        else :
            print("start processing")
            
            # 요청에서 카메라 프레임 데이터를 가져옵니다.
            frame_data = np.frombuffer(request.data, dtype=np.uint8)

        # 프레임 데이터를 처리하기 전에 로그 문장을 추가합니다.
            print('Received camera frame:', len(frame_data), 'bytes')
            process_frame(frame_data)
            print(name)
            

        # 응답으로는 프레임 데이터가 아닌 성공 상태를 반환합니다.
        return 'Success'
    else:
        # 다른 메서드의 경우 오류 응답을 반환합니다.
        return '허용되지 않는 메서드입니다.', 405

@app.route('/join_chatroom')
def join_chatroom():
    return render_template('join_chatroom.html')

def generateRoomID():
    while(True):
        room_id = f'{random.randint(0, 9999):04}'
        if room_id not in rooms:
            return room_id

@socketio.on('join')
def on_join(data):
    username = data['username']
    room_id = data['room_id']

    # 방이 존재하지 않으면 새로 생성
    if room_id == -1:
        room_id = generateRoomID()
        rooms[room_id] = set()
    
    print(room_id)

    # 사용자를 해당 방에 추가하고 방에 조인
    rooms[room_id].add(username)
    join_room(room_id)

    # 해당 방의 사용자 목록을 업데이트한 후 방에 있는 모든 사용자들에게 전송
    emit('update_users', {'room_id': room_id, 'users': list(rooms[room_id])}, room=room_id)


@socketio.on('leave')
def on_leave(data):
    username = data['username']
    room_id = data['room_id']

    # 방에서 사용자 제거하고 방에서 나가기
    if room_id in rooms and username in rooms[room_id]:
        rooms[room_id].remove(username)
        leave_room(room_id)

        # 해당 방의 사용자 목록을 업데이트한 후 방에 있는 모든 사용자들에게 전송
        emit('update_users', {'room_id': room_id, 'users': list(rooms[room_id])}, room=room_id)

        # 방에 사용자가 더 이상 없으면 방을 삭제
        if not rooms[room_id]:
            del rooms[room_id]


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=80, debug=True)