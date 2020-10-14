from cv2 import cv2
import numpy as np
import dlib
from math import hypot
from pynput.mouse import Button,Controller

mouse = Controller()


def midpoint(p1, p2):
    return  (p1.x+p2.x)//2, (p1.y+p2.y)//2

font = cv2.FONT_HERSHEY_SIMPLEX

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x , facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x , facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))
    #hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), thickness=1)
    #ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), thickness=1)
    hor_line_len = hypot((left_point[0] - right_point[0]),(left_point[1] - right_point[1]))
    ver_line_len = hypot((center_top[0] - center_bottom[0]),(center_top[1]-center_bottom[1]))

    ratio = hor_line_len/ver_line_len
    return ratio

def get_gaze_ratio(eye_points, facial_landmarks):
    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x,facial_landmarks.part(eye_points[0]).y),
                                    (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                    (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                    (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                    (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                    (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)
    cv2.polylines(frame,[left_eye_region],True,255,2)

    height, width, _ = frame.shape
    mask = np.zeros((height,width),np.uint8)

    cv2.polylines(mask,[left_eye_region],True,255,2)
    cv2.fillPoly(mask, [left_eye_region],255)

    eye = cv2.bitwise_and(gray,gray,mask=mask)

    min_x = np.min(left_eye_region[:,0])
    max_x = np.max(left_eye_region[:,0])
    min_y = np.min(left_eye_region[:,1])
    max_y = np.max(left_eye_region[:,1])

    gray_eye = eye[min_y:max_y, min_x:max_x]
    _,threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0: height, 0:int(width/2)]
    left_side_white = cv2.countNonZero(left_side_threshold)    

    right_side_threshold = threshold_eye[0:height, int(width/2):width]
    right_side_white = cv2.countNonZero(right_side_threshold)

    up_threshold = threshold_eye[0:int(height/2),0:width]
    up_white = cv2.countNonZero(up_threshold);

    down_threshold=threshold_eye[int(height/2):height, 0:width]
    down_white = cv2.countNonZero(down_threshold);

    if left_side_white==0:
        gaze_ratio=0.5
    elif right_side_white==0:
        gaze_ratio=2
    else:
        gaze_ratio = left_side_white/right_side_white
    if up_white==0:
        vertical_ratio=0.8
    elif down_white==0:
        vertical_ratio=1.2
    else:
        vertical_ratio = up_white/down_white
    return gaze_ratio,vertical_ratio

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    mouse_x, mouse_y = mouse.position

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
        # x ,y = face.left(), face.top()
        # x1, y1 = face.right(), face.bottom()
        # cv2.rectangle(frame, (x ,y), (x1, y1), (255, 0, 0))

        landmarks = predictor(gray, face)
        
        left_eye_ratio = get_blinking_ratio([36,37,38,39,40,41], landmarks)
        right_eye_ratio = get_blinking_ratio([42,43,44,45,46,47],landmarks)

        blinking_ratio = (left_eye_ratio+right_eye_ratio)/2
        if blinking_ratio>5:
            cv2.putText(frame, 'BLINKING',(50,150),font,7,(255,0,0))


        gaze_ratio_left_eye, vertical_ratio_left_eye = get_gaze_ratio([36,37,38,39,40,41],landmarks)
        gaze_ratio_right_eye, vertical_ratio_right_eye = get_gaze_ratio([42,43,44,45,46,47],landmarks)

        gaze_ratio = (gaze_ratio_left_eye+gaze_ratio_right_eye)/2
        vertical_ratio = (vertical_ratio_left_eye+vertical_ratio_right_eye)/2
        if gaze_ratio<0.8:
            cv2.putText(frame,"LEFT",(50,100),font,2,(0,0,255),3)
            mouse_x = mouse_x - 10
            mouse.position = (mouse_x, mouse_y)
        elif 0.8<gaze_ratio<1.1:
            cv2.putText(frame,"CENTER",(50,100),font,2,(0,0,255),3)
        else:
            cv2.putText(frame,"RIGHT",(50,100),font,2,(0,0,255),3)
            mouse_x = mouse_x+ 10
            mouse.position = (mouse_x,mouse_y)

        if vertical_ratio<0.5:
            mouse_y = mouse_y - 10
            mouse.position = (mouse_x, mouse_y)
            cv2.putText(frame,"UP",(50,150),font,2,(0,0,255),3)
        elif 0.5<vertical_ratio<0.8:
            cv2.putText(frame,"CENTER",(50,150),font,2,(0,0,255),3)
        else:
            cv2.putText(frame,"DOWN",(50,150),font,2,(0,0,255),3)
            mouse_y = mouse_y + 10
            mouse.position = (mouse_x,mouse_y)

    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()