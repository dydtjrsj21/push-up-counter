import cv2
import socket
import struct
import pickle
import mediapipe as mp
import json

# Mediapipe 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# POSE_CONNECTIONS을 JSON 직렬화 가능한 형태로 변환
connections = [(conn[0], conn[1]) for conn in mp_pose.POSE_CONNECTIONS]

# 서버 설정
HOST = '0.0.0.0'
PORT = 9999

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)
print(f"서버가 {PORT} 포트에서 수신 대기 중입니다...")

conn, addr = server_socket.accept()
print(f"클라이언트 {addr} 연결됨.")

data = b""
payload_size = struct.calcsize(">L")

# Mediapipe Pose 사용
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    try:
        while True:
            # 프레임 데이터 수신
            while len(data) < payload_size:
                data += conn.recv(4096)

            packed_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack(">L", packed_size)[0]

            while len(data) < msg_size:
                data += conn.recv(4096)

            frame_data = data[:msg_size]
            data = data[msg_size:]

            # 프레임 디코딩
            frame = pickle.loads(frame_data)

            # Mediapipe로 포즈 추정
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # 키포인트 좌표 추출
            landmarks = []
            if results.pose_landmarks:
                for landmark in results.pose_landmarks.landmark:
                    landmarks.append({
                        "x": landmark.x,
                        "y": landmark.y,
                        "z": landmark.z,
                        "visibility": landmark.visibility
                    })

            # 좌표와 연결 정보를 포함한 데이터 생성
            response_data = {
                "landmarks": landmarks,
                "connections": connections
            }
            response_json = json.dumps(response_data)

            # 데이터 길이와 함께 전송
            conn.sendall(struct.pack(">L", len(response_json)) + response_json.encode('utf-8'))

    except KeyboardInterrupt:
        print("서버 중단")
    finally:
        conn.close()
        server_socket.close()
