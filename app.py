import random
import dlib
import cv2
import numpy as np
import os
import pymysql
from collections import deque
from flask import Flask, make_response, redirect, render_template, Response, jsonify, request, flash, session, url_for
from flask_socketio import SocketIO, join_room, leave_room, emit
import hashlib
from datetime import datetime, timedelta
import jwt
from functools import wraps
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key ="Over_the_Glass"
app.config['JWT_SECRET_KEY'] = 'Over_the_Glass'
# 이미지 업로드 폴더 설정 
app.config['UPLOAD_FOLDER'] = './faces' 
# 허용된 파일 확장자 설정
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}  
socketio = SocketIO(app)

# 각자 데이터베이스에 맞춰서 변경 
# db = pymysql.connect(host='localhost', user='root', password='2023', db='overtheglass')
# db = pymysql.connect(host='localhost', user='root', password='0717', db='overtheglass')
db = pymysql.connect(host='localhost', user='root', password='0000', db='overtheglass')

m = hashlib.sha256()
m.update('Over the Glass'.encode('utf-8'))

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
latest_speaker_position = []
difference = [0 for _ in range(len(known_names))]
name = "Unknown"

def process_frame(data):
    global name
    global latest_speaker_position

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
                latest_speaker_position = [face.left(), face.bottom() + 60]
        
        else:  # 매치율이 0.5보다 작을 경우
            name = "Unknown"  # Unknown으로 표시

    if not has_speaker and client_speech:
        if not faces:  # 얼굴이 감지되지 않을 경우
            name = "Unknown"
    
    print("end processing")

    return

@app.route('/')
def main():
    return render_template('new_main.html')

# token을 decode하여 반환, 실패 시 payload = None
def check_access_token(access_token):
    try:
        payload = jwt.decode(access_token, app.config['JWT_SECRET_KEY'] ,'HS256')
        expiration_time = datetime.fromtimestamp(payload['exp'])
        # 토큰 만료된 경우
        if expiration_time < datetime.utcnow():
            payload = None
            
    except jwt.InvalidTokenError:
        payload=None
    
    return payload

# decorator 함수
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwagrs):
        if "token" not in request.cookies:
            return jsonify({"error": "No token in cookies"}), 401
        
        # 요청 토큰 정보 받아오기
        access_token = request.cookies.get('token') 
        payload = check_access_token(access_token)
        #print("access_token", access_token)
        
        if payload is None:
            return jsonify({'error': 'Invalid token'}), 401
        
        return f(payload,*args, **kwagrs)

    return decorated_function

@app.route('/login', methods=['GET'])
def login():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login_process():
    data = request.form
    email = data['userEmail']
    pw = data['userPassword1']
    
    # 모든 정보가 입력되었는지 확인     
    # 이메일 혹은 비밀번호 미입력 시 바로 에러 응답을 반환
    if not email or not pw:
        return jsonify({'error': 'Information not entered 입력되지 않은 정보가 있습니다'}), 400
        
    with db.cursor() as cursor:
        # DB에 입력된 이메일과 일치하는 사용자 정보 조회
            query = "SELECT * FROM user WHERE email = %s"
            cursor.execute(query, (email,))
            user = cursor.fetchone()

            if user:           
                # 회원가입과 같은 방법으로 pw를 암호화
                pw_hash =hashlib.sha256(pw.encode('utf-8')).hexdigest()
                
                # DB에 저장된 해시된 패스워드는 user의 3번 인덱스에 위치
                stored_hashed_pw = user[5] 
                # 0 user_pkey 1 name 2 email 3 pwd_hash 4 subtitle 
                user_pkey = user[0]
                name = user[1]
                email = user[2]
                subtitle = user[4]
                #print("app.py line 192",user_pkey, name, email, subtitle)
                    
                # DB에서 조회한 해시된 패스워드와 입력된 패스워드를 비교
                if pw_hash == stored_hashed_pw:
                    payload = {
                        'user_pkey': user_pkey,
                        'name': name,
                        'email': email,
                        'subtitle': subtitle,
                        'exp': datetime.utcnow() + timedelta(seconds=60*60*24) # 만료 24 hour
                    }
                    # 토큰 생성 
                    token = jwt.encode(payload, app.config['JWT_SECRET_KEY'] ,'HS256')
                    return jsonify({'message': 'Login successful 로그인 완료되었습니다.', 'access_token':token}), 200
                else:                  
                    return jsonify({'error': 'Not match password 비밀번호가 일치하지 않습니다.'}), 401
                
            else:
                # 사용자 없음
                return jsonify({'error': 'No user 회원이 존재하지 않습니다.'}), 404

    #resp = make_response(render_template('menu.html'))
    #resp.set_cookie('info', email)
    #return resp

@app.route('/signup', methods=['GET'])
def signup():
    return render_template('signup.html')

@app.route('/signup', methods=['POST'])
def signup_process():
    try:
        data = request.form
        username = data['userName']
        email = data['userEmail']
        pwd1 = data['userPassword1']
        pwd2 = data['userPassword2']
        sub = request.form.getlist('subtitle') 
        sub_value = 1
        
        """
        print(type(sub))
        print(data)
        print(username, email, pwd1, pwd2, sub)
        
        # 모든 데이터 읽어오기
        select_query = "select * from user"
        with db.cursor() as cursor:
            cursor.execute(select_query)
            result = cursor.fetchall()
            print(result)
        """
        
        # 모든 정보가 입력되었는지 확인 
        if (username and email and pwd1 and pwd2):            
            with db.cursor() as cursor:
            
                # DB에 같은 이메일을 가진 회원이 있는지 확인
                query = "SELECT * FROM user WHERE email=%s"
                cursor.execute(query, (email,))
                existing_user = cursor.fetchone()
                if existing_user:
                    return jsonify({'error': 'Email already in use 이미 사용 중인 이메일입니다.'})
                
                # 비밀번호 일치 확인
                if pwd1 != pwd2:
                    return jsonify({'error': 'The password does not match 비밀번호가 일치하지 않습니다.'}), 400

                # 자막 사용 여부 1(default)
                # When unchecked: sub = [], checked: sub = ['1']
                if not sub: 
                    sub_value = 0
                
                # 비밀번호 해시  sha256 방법(=단방향 암호화. 풀어볼 수 없음)
                pw_hash = hashlib.sha256(pwd1.encode('utf-8')).hexdigest()
                
                # 새로운 사용자 추가
                insert_query = "INSERT INTO user (name, email, pwd_hash, subtitle) VALUES (%s, %s, %s, %s)"
                cursor.execute(insert_query, (username, email, pw_hash, sub_value))
                db.commit()
                return jsonify({'message': 'Sign-up successful 회원가입이 완료되었습니다.'}), 200
            
        else:
            return jsonify({'error': 'Information not entered 입력되지 않은 정보가 있습니다'})

    except Exception as e:
        print(f'회원가입 중 오류 발생: {e}')
        db.rollback()
        return jsonify({'error': 'sign-up failed'}), 500
        

@app.route('/chatroom')
@login_required
def chatroom(payload):
    if payload:
        print("chatroom(payload), @login_required",payload)
        user_pkey = payload.get('user_pkey')
        name = payload.get('name')
        return render_template('chatroom.html',  user_pkey=user_pkey, name=name)
    else:
        return "Error", 401   

@app.route('/video_mode')
def video_mode():
    return render_template('video_mode.html')   

@app.route('/nonmember_menu')
def nonmember_menu():
    return render_template('nonmember_menu.html') 

@app.route('/nonmember_settings')
def nonmember_settings():
    return render_template('nonmember_settings.html')

@app.route('/photo')
def photo():
    return render_template('photo.html')  

@app.route('/menu')
@login_required
def menu(payload):
    if payload:
        #print("menu(payload), @login_required",payload)
        name = payload.get('name')
        subtitle = payload.get('subtitle')
        if subtitle == 0:
            #print("menu(payload), @login_required",name, subtitle)
            return render_template('nonmember_menu.html', name=name)
        else:
            #print("menu(payload), @login_required",name, subtitle)
            return render_template('menu.html', name=name)
    else:
        return "Error", 401

@app.route('/logout', methods=['GET'])
def logout():
    # 로그아웃: 토큰 만료 등의 작업 수행
    # 예를 들어, 토큰을 만료시키거나 세션을 무효화할 수 있습니다.
    
    # 로그아웃 후, 클라이언트에게 로그인 페이지로 이동하는 응답을 보냅니다.
    response = make_response(redirect(url_for('main')))  # 로그인 페이지로 리다이렉트
    response.delete_cookie('token')  # 토큰을 쿠키에서 제거
    return response

@app.route('/record')
def record():
    return render_template('record.html')

@app.route('/history_page')
def history_page():
    return render_template('history_page.html')

@app.route('/setting')
def settings():
    return render_template('settings.html')

@app.route('/speaker_info')
def speaker_info():
    return jsonify({'speaker': name})

@app.route('/process_speech', methods=['POST'])
def process_speech():
    global client_speech
    data = request.get_json()
    client_speech = data['speech']
    speaker_name = data['speakerName']
    room_code = data['room_code']

    print("/process_speech line 387", client_speech, speaker_name, room_code)
    return jsonify(".")

@app.route('/camera', methods=['POST'])
def camera():
    if request.method == 'POST':
        # 요청에서 카메라 프레임 데이터를 가져옵니다.
        frame_data = np.frombuffer(request.data, dtype=np.uint8)

        # 프레임 데이터를 처리하기 전에 로그 문장을 추가합니다.
        print('Received camera frame:', len(frame_data), 'bytes')
        process_frame(frame_data)
        print(name, latest_speaker_position)

        print("emit send_data")
        socketio.emit(
            'send_data',
            {
                'speaker': name,
                'positionX': latest_speaker_position[0],
                'positionY': latest_speaker_position[1]
            }
        )

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
    
    # 자막 사용자가 방을 만들 때만 user_pkey 값이 존재한다 그래서 get 이용
    user_pkey = data.get('user_pkey') 

    print("on_join line 435", username, room_id, user_pkey)
    # 자막 사용자가 대화방 생성 시
    if room_id == -1:
        room_id = generateRoomID()
        rooms[room_id] = set()
        #print("username", username) 
        #print("room_id",room_id)

        # 사용자를 해당 방에 추가하고 방에 조인
        rooms[room_id].add(username)
        join_room(room_id)
        print("rooms", rooms) 

        # 해당 방의 사용자 목록을 업데이트한 후 방에 있는 모든 사용자들에게 전송
        emit('update_users', {'room_id': room_id, 'users': list(rooms[room_id])}, room=room_id)
        
        # user_pkey 값이 있는 자막 사용자라면, DB 데이터 넣기
        if user_pkey:
            try:
                with db.cursor() as cursor:
                    query = "INSERT INTO chatroom(user_fkey, room_num) values (%s, %s) "
                    cursor.execute(query, (user_pkey, room_id))
                    
                    # 방금 삽입한 레코드의 chatroom_pkey 값을 가져옴
                    cursor.execute("SELECT LAST_INSERT_ID()")
                    chatroom_pkey = cursor.fetchone()[0]
                    
                    # 커밋
                    db.commit()
                    
                    print("chatroom_pkey:", chatroom_pkey)
                    # chatroom_pkey 값을 클라이언트에게 보냄
                    emit('chatroom_pkey', {'chatroom_pkey': chatroom_pkey}, room=room_id)
                    
            except Exception as e:
                    # 오류 발생 시 롤백
                    print("오류 발생:", str(e))
                    db.rollback()

    # 존재하지 않는 대화방 코드 입력 시
    elif room_id not in rooms:
        print("wrong room code")
        emit('error', {'message': '존재하지 않는 대화방 코드입니다.'})
    # 올바른 대화방 코드 입력 시
    else:
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

# 이미지 확장자 확인 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/upload', methods=['POST'])
@login_required
def upload(payload):
    image = request.files['image']
    # 이미지 업로드 확인 
    if not image:
        return jsonify({'error': 'No image uploaded'}), 400
    # 이미지 확장자 확인 
    if not allowed_file(image.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    #print(image.filename)
    # image.filename에서 확장자 추출
    file_extension = image.filename.split('.')[-1].lower()
    #print(file_extension)
    #print('app.py line 496', filename)
    if payload:
        filename = payload.get('name')+"."+file_extension
        #print('app.py line 499', filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'],filename)
        image.save(image_path)
        return jsonify({'message':'Image save successful 이미지 폴더 저장 성공'})
    # 이미지 mysql에 저장 코드 추가 예정


@app.route('/auth/kakao/callback')
def kakao_login():
    return render_template('menu.html')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=80, debug=True)