import cv2
import torch
import numpy as np
from collections import deque
from threading import Thread
import time
from tqdm import tqdm

from models.model import PhysNetUpsample
from utils.ppg_process_common_function import postprocess, img_process
from utils.util import load_model

DETECT_FPS_COUNT = 30  # video FPS
TRACK_COUNT_PRE = 5  # tracks per second
RPPG_CAL_STEP = 3  # rppg calulate time step, every 3 seconds
FRAME_LEN = 240  # rppg calulate frame length, model input length
START_INFRENCE_FRAME = 270  # start calulate rppg frame count, START_INFRENCE_FRAME >= FRAME_LEN + START_FRAME
START_FRAME = 10  # start track frame index


def cxy_wh_2_rect(pos, sz):
    return [float(max(float(0), pos[0] - sz[0] / 2)), float(max(float(0), pos[1] - sz[1] / 2)), float(sz[0]),
            float(sz[1])]  # 0-index


def to_input(x):
    x = torch.from_numpy(x)
    x = x.permute(3, 0, 1, 2)  # (frame_length, face_h, face_w, 3) -> (3, frame_length, face_h, face_w)
    x = x.unsqueeze(dim=0)  # (3, frame_length, face_h, face_w) -> (batchsize, 3, frame_length, face_h, face_w)
    x = x.cuda()
    return x


def to_numpy(x):
    return x[0,].cpu().detach().numpy()


def inference(face_queue, method='dft'):
    global hr_predict
    time_predict = time.time()
    face_list = list(face_queue)
    face_list = [img_process(roi) for roi in face_list]

    cur_faces = np.array(face_list)  # (frame_length, face_h, face_w, 3)
    cur_faces = to_input(cur_faces)

    output_wave = rppg_net(cur_faces)
    wave_predict = to_numpy(output_wave)
    hr_predict = postprocess(wave_predict, fps=DETECT_FPS_COUNT, length=FRAME_LEN, method=method)
    time_predict = time.time() - time_predict
    print('predict time: ', round(time_predict * 1000, 2), 'ms')
    print("fps: ", DETECT_FPS_COUNT, "hr prdict: ", round(hr_predict, 2))
    print("-----------------")


def track(frame):
    global location
    ok, bbox = face_tracker.update(frame)
    x, y, w, h = bbox
    target_pos = np.array([x + w / 2, y + h / 2])
    target_sz = np.array([w, h])
    location = cxy_wh_2_rect(target_pos, target_sz)


def run(video_path):
    global hr_predict, location
    face_queue = deque(maxlen=FRAME_LEN)
    frame_count = 0

    video_path = eval(video_path) if video_path.isnumeric() else video_path  # i.e. input_video = '0' local webcam
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_FPS, DETECT_FPS_COUNT)
    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == 0:
                cap.release()
                break
            frame_count = frame_count + 1
            if frame_count < START_FRAME:
                continue

            if frame_count == START_FRAME:
                frame_track = frame.copy()
                cv2.putText(frame, 'Select target face ROI and press ENTER', (20, 30),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            1, (0, 0, 0), 1)

                bbox = cv2.selectROI("demo", frame, fromCenter=False)
                x, y, w, h = bbox
                target_pos = np.array([x + w / 2, y + h / 2])
                target_sz = np.array([w, h])
                face_tracker.init(frame_track, bbox)
                location = cxy_wh_2_rect(target_pos, target_sz)
                pbar = tqdm(total=FRAME_LEN)
                pbar.set_description("Get inference data: ")
            # tracking
            if frame_count > START_FRAME and frame_count % int(DETECT_FPS_COUNT / TRACK_COUNT_PRE) == 0:
                frame_track = frame.copy()
                t_track = Thread(target=track, args=(frame_track, ))
                t_track.start()
                t_track.join()
            # calulate rppg
            if frame_count >= START_INFRENCE_FRAME and frame_count % (DETECT_FPS_COUNT * RPPG_CAL_STEP) == 0:
                print("start inference ...")
                t_rppg = Thread(target=inference, args=(face_queue, ))
                t_rppg.start()
                t_rppg.join()
            if location is not None:
                x_left_top, y_left_top, x_right_bottom, y_right_bottom = \
                    int(location[0]), int(location[1]), int(location[0] + location[2]), int(
                        location[1] + location[3])
                face_roi = frame[y_left_top:y_right_bottom, x_left_top:x_right_bottom, :]
                face_roi = cv2.resize(face_roi, (128, 128))
                if frame_count > START_FRAME:
                    face_queue.append(face_roi)
                    if len(face_queue) < FRAME_LEN:
                        pbar.display()
                        pbar.update(1)
                    else:
                        pbar.close()

                cv2.rectangle(frame, (x_left_top, y_left_top), (x_right_bottom, y_right_bottom), (0, 0, 255))

                if hr_predict is None:
                    text = 'HR predict: Detecting'
                else:
                    text = 'HR predict: ' + str(round(hr_predict, 2))
                cv2.putText(frame, text, (20, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            1, (0, 0, 0), 1)
                cv2.imshow("img", frame)
                cv2.waitKey(1)


def main():
    global rppg_net, hr_predict, face_tracker, location
    hr_predict = None
    location = None

    # 0. set path
    video_path = '0'
    rppg_model_path = 'PhysNet_with_ubfc_physformerloss.pth'

    # 1. build tracker
    face_tracker = cv2.TrackerKCF_create()
    # 2. build rPPG net
    rppg_net = PhysNetUpsample()
    rppg_net = load_model(rppg_net, rppg_model_path)

    # 3. run
    run(video_path)


if __name__ == '__main__':
    main()
