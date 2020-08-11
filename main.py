from cv2 import cv2
import numpy as np
import dlib
from math import hypot


def midpoint(p1, p2):
    return  (p1.x+p2.x)//2, (p1.y+p2.y)//2

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
blinkcount = 0

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
        left_point_1 = (landmarks.part(36).x , landmarks.part(36).y)
        right_point_1 = (landmarks.part(39).x , landmarks.part(39).y)
        left_point_2 = (landmarks.part(42).x , landmarks.part(42).y)
        right_point_2 = (landmarks.part(45).x , landmarks.part(45).y)

        center_top_1 = midpoint(landmarks.part(37), landmarks.part(38))
        center_bottom_1 = midpoint(landmarks.part(41), landmarks.part(40))
        center_top_2 = midpoint(landmarks.part(43), landmarks.part(44))
        center_bottom_2 = midpoint(landmarks.part(47), landmarks.part(46))

        hor_line_1 = cv2.line(frame, left_point_1, right_point_1, (0, 255, 0), thickness=1)
        ver_line_1 = cv2.line(frame, center_top_1, center_bottom_1, (0, 255, 0), thickness=1)
        hor_line_2 = cv2.line(frame, left_point_2, right_point_2, (0, 255, 0), thickness=1)
        ver_line_2 = cv2.line(frame, center_top_2, center_bottom_2, (0, 255, 0), thickness=1)

        hor_line_1_len = hypot((left_point_1[0] - right_point_1[0]),(left_point_1[1] - right_point_1[1]))
        ver_line_1_len = hypot((center_top_1[0] - center_bottom_1[0]),(center_top_1[1]-center_bottom_1[1]))

        hor_line_2_len = hypot((left_point_2[0] - right_point_2[0]),(left_point_2[1] - right_point_2[1]))
        ver_line_2_len = hypot((center_top_2[0] - center_bottom_2[0]),(center_top_2[1]-center_bottom_2[1]))

        ratio_1 = hor_line_1_len/ver_line_1_len
        ratio_2 = hor_line_2_len/ver_line_2_len

        if ratio_1 > 5 and ratio_2>5:
            blinkcount=blinkcount+1
            print("Blinked "+str(blinkcount)+" times")


    
    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()