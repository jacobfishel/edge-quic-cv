import cv2
import pickle
import struct
import socket
from cvzone.FaceDetectionModule import FaceDetector

detector = FaceDetector()

#change to server ip address
server_ip = '192.168.4.54'
port = 5005

received_data = b""
payload_size = struct.calcsize("L")

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

server_socket.bind((server_ip,port))

server_socket.listen(10)


client_socket , client_address = server_socket.accept()
print(f'[*] Accepted connection from {client_address}')

while True:

    while len(received_data) < payload_size:
        received_data += client_socket.recv(4096)
    
    packed_msg_size = received_data[:payload_size]
    received_data = received_data[payload_size:]
    msg_size = struct.unpack("L", packed_msg_size) [0]


    while len(received_data) < msg_size:
        received_data += client_socket.recv(4096)
    

    frame_data = received_data[:msg_size]
    received_data = received_data[msg_size:]


    received_frame = pickle.loads(frame_data)

    received_frame, bboxs = detector.findFaces(received_frame,draw=True)

    cv2.imshow('Client video', received_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
server_socket.close()