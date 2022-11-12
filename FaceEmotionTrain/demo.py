from model_factory import model_build
import torch
import cv2
import numpy as np
import cv2
import math
import pdb
from skimage import transform as trans
import argparse

import yaml
import cv2
import numpy as np
import torch
from insightface.app import FaceAnalysis


def estimate_norm(lmk, arcface_src, image_size=112):
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = float('inf')
    if image_size == 112:
        src = arcface_src
    else:
        src = float(image_size) / 112 * arcface_src

    for i in np.arange(src.shape[0]):
        tform.estimate(lmk, src[i])
        M = tform.params[0:2, :]
        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - src[i])**2, axis=1)))
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i
    return min_M, min_index

def norm_crop(img, kpss, arcface_src, image_size=112):
    M, pose_index = estimate_norm(kpss, arcface_src, image_size)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped

def crop_n_align(app, img, box=False):
    arcface_src = np.array(
        [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
         [41.5493, 92.3655], [70.7299, 92.2041]],
        dtype=np.float32)

    arcface_src = np.expand_dims(arcface_src, axis=0)
    faces = app.get(img)
    find = (len(faces) != 0)
    face_images = []
    bboxs = []
    if find:
        for face in faces:
            kpss = face['kps']
            face_images += [norm_crop(img, kpss, arcface_src, image_size=224)]
            if box:
                bboxs += [face['bbox']]

    if box:
        return face_images, find, bboxs
    else:
        return face_images, find


def demo(cfg, args, mode):
    emo_idx = ["Neutral", "Happiness", "Sadness", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]
    '''
    Test code for face blur detection
    
    Args:
        cfg: configuration file of yaml format
        args
        mode: inference mode. it can be "video" or "image"
    '''
    ##############################
    #       BUILD MODEL          #
    ##############################
    model = model_build(model_name=cfg['train']['model'], num_classes=8, in_channel=3)
    # only predict blur regression label -> num_classes = 1
    if '.ckpt' or '.pt' in args.pretrained_path:
        model_state = torch.load(args.pretrained_path, map_location="cpu")["model_state_dict"]
        model.load_state_dict(model_state)

    device = args.device
    if 'cuda' in device and torch.cuda.is_available():
        model = model.to(device)
        model.eval()
    
    ##############################
    #       MODE : VIDEO         #
    ##############################
    if mode == 'video':
        video_path = args.file_path
        cap = cv2.VideoCapture(video_path, apiPreference=cv2.CAP_MSMF)
        width  = int(cap.get(3))
        height = int(cap.get(4))
        pdb.set_trace()
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        save_path = args.save_path
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

        # for face detection
        app = FaceAnalysis(allowed_modules=['detection'],
                                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))

        while(cap.isOpened()):
            grabbed, frame = cap.read()
            if not grabbed:
                break

            pad = 0
            find = False
            
            while pad <= 200:
                padded = np.pad(frame, ((pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)
                face_images, find, bboxs = crop_n_align(app, padded, box=True)
                if find:
                    break
                pad += 50

            if find:
                for i, face_image in enumerate(face_images):
                    face_image = torch.Tensor(face_image).permute(2, 0, 1).unsqueeze(0)
                    with torch.no_grad():
                    	prediction = model(face_image)
                    _, pred = torch.max(prediction, axis=1)
                    emo_label = emo_idx[int(pred)] # predict blur label

                    bbox = bboxs[i]
                    left_top = (int(bbox[0]-pad//2), int(bbox[1]-pad//2))
                    right_btm = (int(bbox[2]-pad//2), int(bbox[3]-pad//2))
                    red_color = (0, 0, 255)
                    thickness = 3
                    cv2.rectangle(frame, left_top, right_btm, red_color, thickness)

                    TextPosition = (int(bbox[0]), int(bbox[1]))
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 0.5
                    fontColor = (255,255,255)
                    thickness = 1
                    lineType = 2

                    cv2.putText(frame, emo_label,
                        TextPosition,
                        font,
                        fontScale,
                        fontColor,
                        thickness,
                        lineType)
            
            cv2.imshow('emotion image', frame)
            out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

    ##############################
    #       MODE : IMAGE         #
    ##############################
    if mode == 'image':
        image_path = args.file_path
        frame = cv2.imread(image_path)
        width, height = frame.shape[0], frame.shape[1]
        app = FaceAnalysis(allowed_modules=['detection'],
                            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))

        pad = 0
        find = False

        while pad <= 200:
            padded = np.pad(frame, ((pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)
            face_images, find, bboxs = crop_n_align(app, padded, box=True)
            if find:
                break
            pad += 50

            if find:
                for i, face_image in enumerate(face_images):
                    face_image = torch.Tensor(face_image).permute(2, 0, 1).unsqueeze(0)
                    with torch.no_grad():
                    	prediction = model(face_image)
                    _, pred = torch.max(prediction, axis=1)
                    emo_label = emo_idx[int(pred)] # predict blur label

                    bbox = bboxs[i]
                    left_top = (int(bbox[0]-pad//2), int(bbox[1]-pad//2))
                    right_btm = (int(bbox[2]-pad//2), int(bbox[3]-pad//2))
                    red_color = (0, 0, 255)
                    thickness = 3
                    cv2.rectangle(frame, left_top, right_btm, red_color, thickness)

                    TextPosition = (int(bbox[0]), int(bbox[1]))
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 0.5
                    fontColor = (255,255,255)
                    thickness = 1
                    lineType = 2

                    cv2.putText(frame, emo_label,
                        TextPosition,
                        font,
                        fontScale,
                        fontColor,
                        thickness,
                        lineType)

        cv2.imshow('emotion image', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/config_example.yaml', help='Path for configuration file')
    parser.add_argument('--device', type=str, default='cuda', help='Device for model inference. It can be "cpu" or "cuda" ')
    parser.add_argument('--pretrained_path', type=str, default='checkpoint/CHECKPOINT.ckpt', help='Path for pretrained model file')
    parser.add_argument('--mode', type=str, default='video', help='Inference mode. it can be "video" or "image"')
    parser.add_argument('--file_path', type=str, default='./sample.mp4', help='Path for the video or image you want to infer')
    parser.add_argument('--save_path', type=str, default='./output.mp4', help='Path for saved the inference video')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    mode = args.mode.lower()
    assert mode in ['video', 'image']

    demo(cfg, args, mode)
