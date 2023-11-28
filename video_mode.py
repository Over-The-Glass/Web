from collections import deque
import dlib
import cv2
import numpy as np

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

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

known_faces = []
known_names = []

movements = [LipMovement(known_names[i]) for i in range(len(known_names))]
latest_speaker_position = []
difference = [0 for _ in range(len(known_names))]
name = "Unknown"

def initialize_variables():
    global known_faces, known_names, movements, latest_speaker_position, difference
    known_faces = []
    known_names = ["person1"]
    movements = []
    latest_speaker_position = []
    difference = []

def updateDifference():
    global difference
    difference.append(0)

# 변수 초기화
initialize_variables()

def process_video_frame(data):
    global name
    global latest_speaker_position

    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    frame = cv2.flip(image, 0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    has_speaker = False
    landmarks = None

    if not faces:  # 얼굴이 감지되지 않을 경우
        name = "?"  # Unknown으로 표시

    for face in faces:
        landmarks = predictor(gray, face)

        face_encoding = np.array(facerec.compute_face_descriptor(frame, landmarks))

        if not known_faces:
            known_faces.append(face_encoding)
            movements.append(LipMovement("person1"))
            updateDifference()
            print("found new face", known_names[0])
            return known_names[0], [0, 0]

        face_distances = []
        for known_face in known_faces:
            face_distances.append(np.linalg.norm(known_face - face_encoding))

        min_distance = min(face_distances)
        min_distance_index = np.argmin(face_distances)
        name = known_names[min_distance_index]
        match_rate = 1 / (1 + min_distance)

        if match_rate > 7:
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
                latest_speaker_position = [face.left(), face.bottom() + 60]
        
        else:  # 매치율이 0.5보다 작을 경우 새로운 얼굴 정보 등록
            new_name = 'person{}'.format(len(known_faces) + 1)
            known_names.append(new_name)
            known_faces.append(face_encoding)
            movements.append(LipMovement(new_name))
            updateDifference()
            print("found new face", new_name)

    if not has_speaker:
        name = "Unknown"
    
    print("end processing")

    return name, latest_speaker_position