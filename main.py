from cv2 import cv2
import numpy as np
import dlib
from math import hypot


def midpoint(p1, p2):
    return  (p1.x+p2.x)//2, (p1.y+p2.y)//2

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def get_blinking_ratio(eye_points, facial_landmarks):
        left_point = (facial_landmarks.part(eye_points[0]).x , facial_landmarks.part(eye_points[0]).y)
        right_point = (facial_landmarks.part(eye_points[3]).x , facial_landmarks.part(eye_points[3]).y)

        center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
        center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))


        hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), thickness=1)
        ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), thickness=1)

        hor_line_len = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
        ver_line_len = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

        ratio = hor_line_len/ver_line_len


        return ratio


while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
        # x ,y = face.left(), face.top()
        # x1, y1 = face.right(), face.bottom()
        # cv2.rectangle(frame, (x ,y), (x1, y1), (255, 0, 0))

        landmarks = predictor(gray, face)

        left_eye_ratio = get_blinking_ratio([36,37,38,39,40,41], landmarks)


        if left_eye_ratio > 5.39:
            print("Blinked")

        


    
    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()