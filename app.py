import dlib
import cv2
import numpy as np
import os
from collections import deque
from flask import Flask, render_template, Response, jsonify, request

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

def draw_text(img, text, font=cv2.FONT_HERSHEY_SIMPLEX, pos=(0, 0), font_scale=1, font_thickness=2,
              text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), bg_color, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)
    return

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

video_capture = cv2.VideoCapture(0)

def process_frame(frame):
    global latest_speaker_position
    global name
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    has_speaker = False
    landmarks = None

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
            text = f"w: {width_average}, h: {height_average}"

            if height_average > 3 or width_average > 3:
                cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 1)
                difference[min_distance_index] = height_average + width_average
                has_speaker = True

            if np.argmax(difference) == min_distance_index:
                latest_speaker_position = (face.left(), face.bottom() + 60)

            if latest_speaker_position:
                draw_text(frame, client_speech, pos=latest_speaker_position)

            cv2.putText(frame, f"{name} ({match_rate:.2%})", (face.left(), face.bottom() + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    if not has_speaker and client_speech:
        draw_text(frame, client_speech, pos=(frame.shape[1] // 2, frame.shape[0] - 30))

    ret, buffer = cv2.imencode('.jpg', frame)
    frame = buffer.tobytes()
    return frame

def gen():
    global client_speech
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        if check_frame:
            processed_frame = process_frame(frame)

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + processed_frame + b'\r\n')

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

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/process_speech', methods=['POST'])
def process_speech():
    global client_speech
    client_speech = request.json['speech']
    return 'OK'

if __name__ == '__main__':
    app.run(threaded=True)
