import torch
import mmcv
import logging
import torch.multiprocessing as mp
import numpy as np
import cv2
import os
import scipy
from time import time
from mmcv.utils import ProgressBar
from .pose_demo import inference as pose_inference
from mmskeleton.apis.estimation import inference_pose_estimator, init_pose_estimator
from mmskeleton.utils import call_obj, load_checkpoint

action_class = {
0: 'ApplyEyeMakeup',
1: 'ApplyLipstick',
2: 'Archery',
3: 'BabyCrawling',
4: 'BalanceBeam',
5: 'BandMarching',
6: 'BaseballPitch',
7: 'Basketball',
8: 'BasketballDunk',
9: 'BenchPress',
10: 'Biking',
11: 'Billiards',
12: 'BlowDryHair',
13: 'BlowingCandles',
14: 'BodyWeightSquats',
15: 'Bowling',
16: 'BoxingPunchingBag',
17: 'BoxingSpeedBag',
18: 'BreastStroke',
19: 'BrushingTeeth',
20: 'CleanAndJerk',
21: 'CliffDiving',
22: 'CricketBowling',
23: 'CricketShot',
24: 'CuttingInKitchen',
25: 'Diving',
26: 'Drumming',
27: 'Fencing',
28: 'FieldHockeyPenalty',
29: 'FloorGymnastics',
30: 'FrisbeeCatch',
31: 'FrontCrawl',
32: 'GolfSwing',
33: 'Haircut',
34: 'Hammering',
35: 'HammerThrow',
36: 'HandstandPushups',
37: 'HandstandWalking',
38: 'HeadMassage',
39: 'HighJump',
40: 'HorseRace',
41: 'HorseRiding',
42: 'HulaHoop',
43: 'IceDancing',
44: 'JavelinThrow',
45: 'JugglingBalls',
46: 'JumpingJack',
47: 'JumpRope',
48: 'Kayaking',
49: 'Knitting',
50: 'LongJump',
51: 'Lunges',
52: 'MilitaryParade',
53: 'Mixing',
54: 'MoppingFloor',
55: 'Nunchucks',
56: 'ParallelBars',
57: 'PizzaTossing',
58: 'PlayingCello',
59: 'PlayingDaf',
60: 'PlayingDhol',
61: 'PlayingFlute',
62: 'PlayingGuitar',
63: 'PlayingPiano',
64: 'PlayingSitar',
65: 'PlayingTabla',
66: 'PlayingViolin',
67: 'PoleVault',
68: 'PommelHorse',
69: 'PullUps',
70: 'Punch',
71: 'PushUps',
72: 'Rafting',
73: 'RockClimbingIndoor',
74: 'RopeClimbing',
75: 'Rowing',
76: 'SalsaSpin',
77: 'ShavingBeard',
78: 'Shotput',
79: 'SkateBoarding',
80: 'Skiing',
81: 'Skijet',
82: 'SkyDiving',
83: 'SoccerJuggling',
84: 'SoccerPenalty',
85: 'StillRings',
86: 'SumoWrestling',
87: 'Surfing',
88: 'Swing',
89: 'TableTennisShot',
90: 'TaiChi',
91: 'TennisSwing',
92: 'ThrowDiscus',
93: 'TrampolineJumping',
94: 'Typing',
95: 'UnevenBars',
96: 'VolleyballSpiking',
97: 'WalkingWithDog',
98: 'WallPushups',
99: 'WritingOnBoard',
100: 'YoYo'
}

def render(image, pred, recog_pred, person_bbox, bbox_thre=0):
    if pred is None:
        return image

    mmcv.imshow_det_bboxes(image,
                           person_bbox,
                           np.zeros(len(person_bbox)).astype(int),
                           class_names=['person'],
                           score_thr=bbox_thre,
                           show=False,
                           wait_time=0)

    for person_pred in pred:
        for joint_pred in person_pred:
            cv2.circle(image, (int(joint_pred[0]), int(joint_pred[1])), 2,
                       [255, 0, 0], 2)

    recog_pred = scipy.special.softmax(recog_pred)
    label_text_0 = f"TOP 1"
    cv2.putText(image, label_text_0, (5, 40),
                cv2.FONT_HERSHEY_SIMPLEX, fontScale=.3, color=[0, 69, 255])
    # label_text_1 = f"label:{action_class[recog_pred.argmax()]} score:{recog_pred.max():.2f}"
    label_text_1 = f"{action_class[recog_pred.argmax()]}"

    cv2.putText(image, label_text_1, (5, 50),
                cv2.FONT_HERSHEY_SIMPLEX, fontScale=.3, color=[0, 140, 255])
    label_text_2 = f"TOP 5"
    cv2.putText(image, label_text_2, (5, 80),
                cv2.FONT_HERSHEY_SIMPLEX, fontScale=.3, color=[0, 69, 255])
    label_text_3 = f"{action_class[np.argsort(recog_pred)[0][-1]]}"
    cv2.putText(image, label_text_3, (5, 90),
                cv2.FONT_HERSHEY_SIMPLEX, fontScale=.3, color=[0, 140, 255])
    label_text_4 = f"{action_class[np.argsort(recog_pred)[0][-2]]}"
    cv2.putText(image, label_text_4, (5, 100),
                cv2.FONT_HERSHEY_SIMPLEX, fontScale=.3, color=[0, 140, 255])
    
    #print(action_class[np.argsort(recog_pred)[0][-2]])
    #try:
    #    print(action_class[np.argsort(recog_pred)[0][-3]])
    #except Exception as e2:
    #    print(e2)

    try:
        label_text_5 = f"{action_class[np.argsort(recog_pred)[0][-3]]}"
        cv2.putText(image, label_text_5, (5, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=.3, color=[0, 140, 255])
    except Exception as e2:
        print("e2= ",e2)

    try:
        label_text_6 = f"{action_class[np.argsort(recog_pred)[0][-4]]}"
        cv2.putText(image, label_text_6, (5, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=.3, color=[0, 140, 255])
    except Exception as e3:
        print("e3= ",e3)

    try:
        label_text_7 = f"{action_class[np.argsort(recog_pred)[0][-5]]}"
        cv2.putText(image, label_text_7, (5, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=.3, color=[0, 140, 255])
    except Exception as e4:
        print("e4= ",e4)
    
    return np.uint8(image)

def init_recognizer(recognition_cfg, device):
    model = call_obj(**(recognition_cfg.model_cfg))
    load_checkpoint(model,
                    recognition_cfg.checkpoint_file,
                    map_location=device)
    return model


def inference(detection_cfg,
              estimation_cfg,
              recognition_cfg,
              video_file,
              gpus=1,
              worker_per_gpu=1,
              save_dir=None):

    recognizer = init_recognizer(recognition_cfg, "cpu")
    # print(np.array(recognizer).shape) # ()
    # import IPython
    # IPython.embed()
    resolution = mmcv.VideoReader(video_file).resolution
    # print(resolution) # (340, 256)
    pose_model = init_pose_estimator(detection_cfg, estimation_cfg, device=0)
    results = pose_inference(detection_cfg, estimation_cfg, video_file, gpus,
                             worker_per_gpu,save_dir=save_dir)
    # print(np.array(results).shape) # (300,)
    
    seq = np.zeros((1, 3, len(results), 17, 1)) # batch size/# videos, inputs/channels, frames, keypoints, # persons in video
    # print(seq)
    # print(seq.shape) # (1, 3, 300, 17, 1)
    for i, r in enumerate(results): # copy of frame by frame of locations and scores
        if r['joint_preds'] is not None:
            seq[0, 0, i, :, 0] = r['joint_preds'][0, :, 0] / resolution[0] #scalation, note: check slices, numpy slicing
            seq[0, 1, i, :, 0] = r['joint_preds'][0, :, 1] / resolution[1]
            seq[0, 2, i, :, 0] = r['joint_scores'][0, :, 0]
    try:
        with torch.no_grad():
            preds = recognizer.double()(torch.tensor(seq).double()).data.cpu().numpy() # preds= recognizer(seq)
            # print(preds.shape) # (1, 101)
            print("\n\nACTION RECOGNITION")
            print("TOP 1: ",preds.argmax())
            print("TOP 5: ",np.argsort(preds)[0][-5:]) # print top five
    except Exception as e:
        print(e)
    # import IPython
    # IPython.embed()
   
    video_frames = mmcv.VideoReader(video_file)
    all_result = []
    print('\n\nAction estimation:')
    prog_bar = ProgressBar(len(video_frames))
    for i, image in enumerate(video_frames):
        res = inference_pose_estimator(pose_model, image)
        inp = torch.tensor(seq[:,:,:i+2,:,:]).double()
        preds = recognizer.double()(inp).data.cpu().detach().numpy()
        
        res['frame_index'] = i
        if save_dir is not None:
            res['render_image'] = render(image, res['joint_preds'], preds,
                                         res['person_bbox'],
                                         detection_cfg.bbox_thre)
        all_result.append(res)
        prog_bar.update()
    # sort results
    all_result = sorted(all_result, key=lambda x: x['frame_index'])

    # generate video
    if (len(all_result) == len(video_frames)) and (save_dir is not None):
        print('\n\nGenerate video:')
        video_name = video_file.strip('/n').split('/')[-1]
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        video_path = os.path.join(save_dir, video_name)
        vwriter = cv2.VideoWriter(video_path,
                                  mmcv.video.io.VideoWriter_fourcc(*('mp4v')),
                                  video_frames.fps, video_frames.resolution)
        prog_bar = ProgressBar(len(video_frames))
        for r in all_result:
            vwriter.write(r['render_image'])
            prog_bar.update()
        vwriter.release()
        print('\nVideo was saved to {}'.format(video_path))
    
    return results

