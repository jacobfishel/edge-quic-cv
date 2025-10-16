import cv2
import pickle
import struct
import socket

cam = cv2.VideoCapture(0)

#change to server ip address
server_ip = 'localhost'
port = 1200

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

client_socket.connect((server_ip, port))

while True:
    ret, frame = cam.read()

    serialized_frame = pickle.dumps(frame)

    message_size = struct.pack("L", len(serialized_frame))

    client_socket.sendall(message_size + serialized_frame)

    cv2.imshow('Client video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
client_socket.close()
