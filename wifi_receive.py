import socket
import cv2
import numpy as np

ESP_IP = "ESP_SERVER_IP_ADDRESS"
ESP_PORT = 12345

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((ESP_IP, ESP_PORT))

cv2.namedWindow("Stream :)", cv2.WINDOW_NORMAL)

while True:
    try:
        frame_data = b''
        while True:
            packet = client_socket.recv(4096)
            if not packet:
                break
            frame_data += packet

        frame_array = np.frombuffer(frame_data, dtype=np.uint8)

        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

        cv2.imshow("Video Stream", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        print(f"Error: {e}")
        break

cv2.destroyAllWindows()
client_socket.close()
