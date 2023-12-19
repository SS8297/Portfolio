import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import os
from twilio.rest import Client
from streamlit_image_select import image_select
import cv2 as cv
import numpy as np
import math
from feat import Detector
from feat.utils import FEAT_EMOTION_COLUMNS
import torch
from PIL import Image

os.environ["TWILIO_ACCOUNT_SID"] = "ACf1e76f3fd6e9cbca940decc4ed443c20"
os.environ["TWILIO_AUTH_TOKEN"] = "56a1d1ee494933269fe042706392ac9f"


def get_ice_servers():
    try:
        account_sid = os.environ["TWILIO_ACCOUNT_SID"]
        auth_token = os.environ["TWILIO_AUTH_TOKEN"]
    except KeyError:
        logger.warning("TURN credentials are not set. Fallback to a free STUN server from Google.")
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

    client = Client(account_sid, auth_token)

    token = client.tokens.create()

    return token.ice_servers

def eye_aspect_ratio(eye):

	A = math.dist(eye[1], eye[5])
	B = math.dist(eye[2], eye[4])

	C = math.dist(eye[0], eye[3])

	ear = (A + B) / (2.0 * C)

	return ear

def detect_eyes(landmarks, img, threshold):
    lm = landmarks
    eyes = np.array(lm[36:48], np.int32)

    left_eye = eyes[0:6]
    right_eye = eyes[6:12]
    ear = max(eye_aspect_ratio(left_eye), eye_aspect_ratio(right_eye))
    left_eye = left_eye.reshape((-1,1,2))
    right_eye = right_eye.reshape((-1,1,2))
    cv.polylines(img, [left_eye], True, (0, 255, 255))
    cv.polylines(img, [right_eye], True, (255, 0, 255))

    if (ear > threshold):
         return True
    else:
         return False

def proc_image(img, detector):
    detected_faces = detector.detect_faces(img)
    faces_detected = len(detected_faces[0])
    if ( faces_detected < 1):
        return img
    
    detected_landmarks = detector.detect_landmarks(img, detected_faces)
    assert len(detected_landmarks[0]) == faces_detected, "Number of faces and landsmarks are mismatched!"

    is_eye_open = [detect_eyes(face, img, 0.20) for face in detected_landmarks[0]]
    eye_dict = {True: "eyes open", False: "eyes closed"}

    detected_emotions = detector.detect_emotions(img, detected_faces, detected_landmarks)
    assert len(detected_emotions[0]) == faces_detected, "Number of faces and emotions are mismatched!"

    em = detected_emotions[0]
    em_labels = em.argmax(axis=1)



    for face, has_open_eyes, label in zip(detected_faces[0], (eye_dict[eyes] for eyes in is_eye_open), em_labels):
        (x0, y0, x1, y1, p) = face
        res_scale = img.shape[0]/704
        cv.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), color = (0, 0, 255), thickness = 3)
        cv.putText(img, FEAT_EMOTION_COLUMNS[label], (int(x0)-10, int(y1+25*res_scale*1.5)), fontFace = 0, color = (0, 255, 0), thickness = 2, fontScale = res_scale)
        cv.putText(img, f"{faces_detected } face(s) found", (0, int(25*res_scale*1.5)), fontFace = 0, color = (0, 255, 0), thickness = 2, fontScale = res_scale)
        cv.putText(img, has_open_eyes, (int(x0)-10, int(y0)-10), fontFace = 0, color = (0, 255, 0), thickness = 2, fontScale = res_scale)
    return img
    
def extract_feat():
    return [1,2,3,4,5]
    
def image_processing(frame):
    return proc_image(img, detector) if recog else img

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    ann = proc_image(img, detector) if recog else img

    return av.VideoFrame.from_ndarray(ann, format="bgr24")

detector = Detector(face_model="retinaface", landmark_model= "pfld", au_model = "xgb", emotion_model="resmasknet")
source = "Webcam"
recog = True

source  = st.radio(
    label = "Image source for emotion recognition",
    options = ["Webcam", "Images"],
    horizontal = True,
    label_visibility = "collapsed",
    args = (source, )
    )

has_cam = True if (source == "Webcam") else False

stream = st.container()
with stream:
    if has_cam:
        webrtc_streamer(
            key="example",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=video_frame_callback,
            rtc_configuration={ "iceServers": get_ice_servers() },
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
    else:
        pic = st.container()
        frame = image_select(
        label="Try the classifier on one of the provided examples!",
        images=[
            "ex1.jpg",
            "ex4.jpg",
            "ex5.jpg",
            "ex6.jpg",
        ],
        use_container_width= False
        )
        img = np.array(Image.open(frame))
        pic.image(image_processing(img), width = 704)


recog = st.toggle(":green[Emotion recogntion]", key = "stream", value = True)

