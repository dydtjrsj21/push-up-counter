import cv2
import socket
import struct
import pickle
import mediapipe as mp

# Mediapipe 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 서버 주소와 포트
HOST = '0.0.0.0'  # 모든 IP에서 수신
PORT = 9999       # 서버 포트 번호

# 소켓 생성 및 바인딩
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)
print(f"서버가 {PORT} 포트에서 수신 대기 중입니다...")

# 클라이언트 연결 수락
conn, addr = server_socket.accept()
print(f"클라이언트 {addr} 연결됨.")

data = b""
payload_size = struct.calcsize(">L")

# Mediapipe Pose 사용
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    try:
        while True:
            # 패킷 크기 수신
            while len(data) < payload_size:
                data += conn.recv(4096)

            packed_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack(">L", packed_size)[0]

            # 실제 데이터 수신
            while len(data) < msg_size:
                data += conn.recv(4096)

            frame_data = data[:msg_size]
            data = data[msg_size:]

            # 프레임 디코딩
            frame = pickle.loads(frame_data)

            # Mediapipe를 사용해 포즈 추정
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # 포즈 랜드마크를 영상 위에 그리기
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )

            # 결과 영상 표시
            cv2.imshow('Server - Pose Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("서버 중단")
    finally:
        conn.close()
        server_socket.close()
        cv2.destroyAllWindows()
