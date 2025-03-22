import cv2
import mediapipe as mp
import time
import torch as t
from .model import HandModel
from .tools.landmark_handle import landmark_handle
from .tools.draw_landmarks import draw_landmarks
from .tools.calc_landmark_list import calc_landmark_list
from .tools.draw_bounding_rect import draw_bounding_rect
import numpy as np
from .tools.draw_rect_text import draw_rect_txt
from .tools.common import labels
import os
import logging

def sign_language_recognition(filepath):
    logging.info("Sign language recognition")
    logging.info(f"Start sign language recognition for {filepath}")
    logging.info(f"is existed: {os.path.exists(filepath)}")

    if not check_video_readability(filepath):
        logging.error(f"Video file cannot be read: {filepath}")
        return {"error": "Video file cannot be processed"}
    else:
        logging.info("Video can be read")

    module_dir = os.path.dirname(__file__)
    model_path = os.path.join(module_dir, 'checkpoints/model_39.pth')
    # model_path = 'checkpoints/model_test1.pth'
    # model_path = 'checkpoints/model_39.pth'

    label = labels
    label_num = len(label)

    background_flag = 0
    background_color = 128

    model = HandModel(num_classes=label_num)
    state_dict = t.load(model_path)
    model.load_state_dict(state_dict)

    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.9,
        min_tracking_confidence=0.9)
    #cap = cv2.VideoCapture(0)
    #cap = cv2.VideoCapture(f"./Video/user/{video_name}.mp4")
    cap = cv2.VideoCapture(filepath)

    words = []
    word_count = {}
    frame_count = 0
    # time1 = time.time()
    fps = 0
    threshold = 0.7
    min_confidence = -500
    word_time = {}
    last_label = None
    min_count = 20

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logging.info("End of video or error reading frame")
            break

        if frame is None or frame.size == 0:
            logging.warning("Empty frame detected, skipping")
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)
        results = hands.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        hand_local = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                for i in range(21):
                    x = min(int(hand_landmarks.landmark[i].x * frame.shape[1]), frame.shape[1] - 1)
                    y = min(int(hand_landmarks.landmark[i].y * frame.shape[0]), frame.shape[0] - 1)
                    hand_local.append([x, y])

                if background_flag:
                    frame = np.zeros(frame.shape, np.uint8)
                    frame.fill(128)

                draw_landmarks(frame, hand_local)
                brect = draw_bounding_rect(frame, hand_local)
                hand_local = landmark_handle(hand_local)
        this_label = ''
        if hand_local:
            input_tensor = t.tensor(hand_local).float().unsqueeze(0)
            output = model(input_tensor)
            index = output.argmax().item()
            value = output[0, index].item()
            this_label = label[index]
            current_time = time.time()
            # if value < min_confidence:
            #     logging.info(f"Confidence too low, skipping -- [{this_label}] -- [{value}]")
            #     continue

            if this_label == last_label:
                if this_label not in word_count:
                    word_count[this_label] = 0

                logging.info(f"Word {this_label} count is {word_count[this_label]}, Confidence: {value}, time: {current_time - word_time[this_label]}")
                if this_label in word_time and (current_time - word_time[this_label]) < threshold:
                    word_count[this_label] += 1

                    if word_count[this_label] >= min_count:
                        logging.info(f"Word {this_label} is greater than {min_count}")
                        if not words or words[-1] != this_label:
                            words.append(this_label)
                            logging.info(f"Words : {words}")
                        else:
                            logging.info(f"Skipping duplicate word: {this_label}")
                        logging.info(f"Words : {words}")
                        word_time.pop(this_label, None)
                        word_count.pop(this_label, None)
                else:
                    word_count[this_label] = 1
            else:
                if last_label and last_label in word_count:
                    word_count.pop(last_label, None)

                word_count[this_label] = 1

            word_time[this_label] = current_time
            last_label = this_label

            #draw_rect_txt(frame, this_label + ":" + str(value), brect)

            # if value > 9:
            #     cv2.putText(frame,
            #                 this_label,
            #                 (30, 50),
            #                 cv2.FONT_HERSHEY_SIMPLEX,
            #                 1.5,
            #                 (255, 255, 255),
            #                 3)

        # time2 = time.time()
        # frame_count += 1
        # if time2 - time1 >= 0.5:
        #     if frame_count > 0:
        #         fps = round(frame_count / (time2 - time1), 2)
        #         time1 = time.time()
        #         frame_count = 0

        # cv2.putText(frame,
        #             str(fps),
        #             (5, 15),
        #             cv2.FONT_HERSHEY_SIMPLEX,
        #             0.5,
        #             (255, 255, 255),
        #             1)

        # cv2.imshow('MediaPipe Hands', frame)
        # if cv2.waitKey(1) & 0xFF == 27:
        #     break
    cap.release()
    logging.info(f"Finished sign language recognition -- {words}")
    return ' '.join(words)

def check_video_readability(video_path):
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    ret, frame = cap.read()
    cap.release()
    return ret and frame is not None