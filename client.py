import cv2
import socket
import struct
import pickle
import json

# 서버 주소와 포트
SERVER_IP = '127.0.0.1'  # 서버 IP 주소
SERVER_PORT = 9999       # 서버 포트 번호

# 소켓 생성 및 연결
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((SERVER_IP, SERVER_PORT))

# 웹캠 열기
cap = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임 직렬화
        data = pickle.dumps(frame)
        size = len(data)

        # 프레임 전송 (길이 + 데이터)
        client_socket.sendall(struct.pack(">L", size) + data)

        # 좌표 데이터 수신
        data = b""
        payload_size = struct.calcsize(">L")

        while len(data) < payload_size:
            data += client_socket.recv(4096)

        packed_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack(">L", packed_size)[0]

        while len(data) < msg_size:
            data += client_socket.recv(4096)

        response_data = data[:msg_size].decode('utf-8')
        response = json.loads(response_data)

        # 랜드마크 및 연결 정보 가져오기
        landmarks = response["landmarks"]
        connections = response["connections"]

        # 랜드마크를 영상 위에 그리기
        for landmark in landmarks:
            x, y = int(landmark['x'] * frame.shape[1]), int(landmark['y'] * frame.shape[0])
            visibility = landmark['visibility']
            if visibility > 0.5:  # 신뢰도 필터링
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # 연결 정보로 스켈레톤 그리기
        for conn in connections:
            start_idx, end_idx = conn
            if (
                landmarks[start_idx]['visibility'] > 0.5 and
                landmarks[end_idx]['visibility'] > 0.5
            ):
                start_point = (
                    int(landmarks[start_idx]['x'] * frame.shape[1]),
                    int(landmarks[start_idx]['y'] * frame.shape[0])
                )
                end_point = (
                    int(landmarks[end_idx]['x'] * frame.shape[1]),
                    int(landmarks[end_idx]['y'] * frame.shape[0])
                )
                cv2.line(frame, start_point, end_point, (255, 0, 0), 2)

        # 결과 영상 표시
        cv2.imshow('Client - Pose Visualization', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("클라이언트 중단")
finally:
    cap.release()
    client_socket.close()
    cv2.destroyAllWindows()
