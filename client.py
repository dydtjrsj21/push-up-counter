import cv2
import socket
import struct
import pickle

# 서버 주소와 포트
SERVER_IP = '127.0.0.1'  # 서버 IP 주소
SERVER_PORT = 9999       # 서버 포트 번호

# 소켓 생성 및 연결
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((SERVER_IP, SERVER_PORT))
connection = client_socket.makefile('wb')

# 웹캠 열기
cap = cv2.VideoCapture(1)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임을 직렬화
        data = pickle.dumps(frame)
        size = len(data)

        # 데이터 송신 (길이 + 데이터)
        client_socket.sendall(struct.pack(">L", size) + data)

except KeyboardInterrupt:
    print("송신 중단")
finally:
    cap.release()
    client_socket.close()
