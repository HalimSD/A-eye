import argparse
import cv2
from utils.clip_synthesized import ClipCaptionPrefix, ClipCaptionModel, generate_beam
from utils.clip_synthesized import generate2, hps, net_g, get_text
import os
import torch
import clip
import cv2 as cv
from transformers import GPT2Tokenizer
from PIL import Image 
import numpy as np
import time 
import sounddevice as sd
import PIL
from collections import OrderedDict
from PIL import Image
import wavio as wv
import simpleaudio as sa
from torch.utils.data import Dataset
from train import ClipCaptionPrefix as transformerClipCaptionPrefix
from skimage import io
from utils import utils
import os
import glob
import torch
import cv2
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
clip_model, preprocess = clip.load('ViT-B/32', device=device)
hps = utils.get_hparams_from_file("./configs/ljs_base.json")
# midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
# midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
# transform = midas_transforms.dpt_transform

def generate_caption(PIL_image, model): 
    start_time = time.time() 
    with torch.no_grad():
        image = preprocess(PIL_image).unsqueeze(0).to(device)
        prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)
        prefix_embed = model.clip_project(prefix).reshape(1, 10, -1)
        captoin = generate2(model, tokenizer, embed=prefix_embed)
    print("--- %s seconds to load model ---" % (time.time() - start_time))
    return captoin   

def read_caption(caption):
    with torch.no_grad():
        start_time = time.time()
        stn_tst = get_text(caption, hps)
        x_tst = stn_tst.unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
        audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.float().numpy()
        # sd.play(audio, blocking=True, samplerate=hps.data.sampling_rate)
        
        wv.write("generated_voice.wav", audio ,rate=hps.data.sampling_rate, sampwidth=1)
        wav_obj = sa.WaveObject.from_wave_file("generated_voice.wav")
        wav_obj.play().wait_done()
        print("--- %s seconds to read the generated captions ---" % (time.time() - start_time))

def caption_from_device (model):
        start_time = time.time()
        test_data_path = os.path.join(os.getcwd(),'data/conceptual/test')
        image_paths = [os.path.join(test_data_path, name) for name in os.listdir(test_data_path) if name[-4] == '.']
        img_list = [Image.open(image) for image in image_paths]    
        for image in img_list:
                image_cv = np.array(image)
                cv.imshow('test', image_cv)
                keypress = cv.waitKey(1)
                # model = args.model
                caption = generate_caption(image, model )
                read_caption(caption)
        print("--- %s overal time ---" % (time.time() - start_time))
       
def screen():
    cap = cv2.VideoCapture(0)
    ret,frame = cap.read()
    frame = PIL.Image.fromarray(frame)
    return frame

def caption_live(model):
    cam = cv.VideoCapture(0)
    while True:
        _, frame = cam.read()
        cv.imshow('video', frame)
        keypress = cv.waitKey(1000)
        if keypress & 0xFF != ord('q'):
            pil_image = PIL.Image.fromarray(frame)
            caption = generate_caption(pil_image, model)
            read_caption(caption)
        elif keypress & 0xFF == ord('q'):
            break
    cam.release()
    cv.destroyAllWindows()
    # cam = cv.VideoCapture(0)
    # while cam.isOpened():
    #     _, frame = cam.read()
    #     # cv.imshow('video', frame)
    #     start = time.time()
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     input_batch = transform(frame).to(device)
    #     with torch.no_grad():
    #         prediction = midas(input_batch)
    #         prediction = torch.nn.functional.interpolate(
    #             prediction.unsqueeze(1),
    #             size=frame.shape[:2],
    #             mode="bicubic",
    #             align_corners=False,
    #         ).squeeze()
    #     depth_map = prediction.cpu().numpy()
    #     depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    #     end = time.time()
    #     totalTime = end - start
    #     fps = 1 / totalTime
    #     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #     depth_map = (depth_map*255).astype(np.uint8)
    #     depth_map = cv2.applyColorMap(depth_map , cv2.COLORMAP_MAGMA)
    #     cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
    #     # cv2.imshow('Image', frame)
    #     cv2.imshow('Depth Map', depth_map)
    #         # h, w = frame.shape[:2]
    #         # center_point = (int(w//2), int(h//2))
    #         # depth_face = depth_map[int(center_point[1]), int(center_point[0])]
    #         # depth = -1.7 * depth_face + 2
    #         # print("Depth in cm: " + str(round(depth,2)*100))
    #         # depth_calc(frame)
    #         # pil_image = PIL.Image.fromarray(frame)
    #         # caption = generate_caption(pil_image, args, model)
    #         # read_caption(caption)
    #     keypress = cv.waitKey(1000)
    #     # if keypress & 0xFF != ord('q'):
    #     if keypress & 0xFF == ord('q'):
    #         break
    # cam.release()
    # cv.destroyAllWindows()

# def depth_calc(img):
#     print(f'img.shape = {img.shape}')
#     input_batch = transform(img).to(device)
#     with torch.no_grad():
#         prediction = midas(input_batch)
#         prediction = torch.nn.functional.interpolate(
#             prediction.unsqueeze(1),
#             size=img.shape[:2],
#             mode="bicubic",
#             align_corners=False,
#         ).squeeze()
#     depth_map = prediction.cpu().numpy()
#     depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
#     depth_map = (depth_map*255).astype(np.uint8)
#     h, w = img.shape[:2]
#     center_point = (int(w//2), int(h//2))
#     depth_face = depth_map[int(center_point[1]), int(center_point[0])]
#     depth = -1.7 * depth_face + 2
#     print("Depth in cm: " + str(round(depth,2)*100))


WEIGHTS_PATHS = {
"project_conceptual": 'checkpoints/conceptual_0',
'project_coco': 'data/coco',
"pretrained_conceptual": 'pretrained_models/cons',
"pretrained_coco": 'pretrained_models',
"pretrained_coco_transformer": 'pretrained_models/v0_models',
}

def last_model (args: argparse.Namespace):

    if args.project and args.coco:
        model_path = WEIGHTS_PATHS.get('project_conceptual')
    elif args.project and args.conceptual:
        model_path = WEIGHTS_PATHS.get('project_conceptual')
        root_cons_models = os.path.join(model_path)
        list_models_path = os.listdir(root_cons_models)
        weights_list = []
        for weight_path in list_models_path:
            if '.pt' in weight_path:
                path = os.path.join(root_cons_models, weight_path)
                print(f'path = {path}')
                weights_list.append(path) 
        latest_model = max(weights_list, key=os.path.getctime)
        return latest_model
    elif args.pretrained and args.conceptual:
        model_path = WEIGHTS_PATHS.get('pretrained_conceptual')
    elif args.pretrained and args.coco and args.transformer:
        model_path = WEIGHTS_PATHS.get('pretrained_coco_transformer')
    elif args.pretrained and args.coco:
        model_path = WEIGHTS_PATHS.get('pretrained_coco')
   
    else:
        assert 'Arguments are not complete'
    


def load_checkpoint(args: argparse.Namespace):
    adjusted_checkpoint = OrderedDict()
    latest_model = last_model(args)
    if args.project and args.conceptual:
        print('Conceptual project model')
        prefix_length = 40
        clip_length = 40
        prefix_size = 512
        checkpoint = torch.load(latest_model, map_location=device)
        for k, v in checkpoint.items():
            if 'clip_project.' in k:
                name = k[13:] # .replace( 'clip_model' , 'clip_project') # remove `module.`
            else:
                name =   k #'clip_model.' +  k
            adjusted_checkpoint[name] = v
        [print(k) for k,_ in adjusted_checkpoint.items()]
        # [print(k) for k,_ in adjusted_checkpoint.items()]
        model = ClipCaptionModel(prefix_length)
        model.load_state_dict(adjusted_checkpoint)
        model.eval()
        model.to(device=device)
        return model  

    elif args.project and args.coco:
        print('Coco project model')
        prefix_length = 40
        clip_length = 40
        prefix_size = 640
        checkpoint = torch.load(latest_model, map_location=device)
        for k, v in checkpoint.items():
            name = k.replace( 'clip_model' , 'clip_project') # remove `module.`
            adjusted_checkpoint[name] = v
        [print(k) for k,_ in adjusted_checkpoint.items()]
        model = ClipCaptionModel(prefix_length)
        model.load_state_dict(adjusted_checkpoint)
        model.eval()
        model.to(device=device)
        return model  

    elif args.pretrained and args.coco and not args.transformer:
        print('Coco pretrained model')
        prefix_length = 10
        clip_length = 10
        prefix_size = 512
        model_path = os.path.join(WEIGHTS_PATHS.get('pretrained_coco'), 'coco_weights.pt')
        checkpoint = torch.load(model_path, map_location=device)
        model = ClipCaptionModel(prefix_length, prefix_size=prefix_size)
        model.load_state_dict(checkpoint)
        model.eval()
        model.to(device=device)
        return model  

    # elif args.pretrained:
    elif args.pretrained and args.conceptual:
        print('Conceptual pretrained model')
        prefix_length = 10
        clip_length = 10
        prefix_size = 512
        model_path = os.path.join(WEIGHTS_PATHS.get('pretrained_conceptual'), 'conceptual_weights.pt')
        checkpoint = torch.load(model_path, map_location=device)
        model = ClipCaptionModel(prefix_length= prefix_length)
        model.load_state_dict(checkpoint)
        model.eval()
        model.to(device=device)
        # midas.to(device)
        # midas.eval()
        return model  

    elif args.transformer and args.pretrained and args.coco:
        print('Transformer based coco pretrained model')
        prefix_length = 40
        clip_length = 40
        prefix_size = 640
        model_path = os.path.join(WEIGHTS_PATHS.get('pretrained_coco_transformer'), 'transformer_weights.pt')
        print(model_path)
        checkpoint = torch.load(model_path, map_location=device)
        for k,v in checkpoint.items():
            if 'clip_project' in k:
                k = k.replace('clip_project', 'clip_model')
        model = transformerClipCaptionPrefix(prefix_length, clip_length=clip_length, prefix_size=prefix_size,
                                  num_layers=8, mapping_type='transformer')

        model.load_state_dict(checkpoint)
        model.eval()
        model.to(device=device)
        return model  

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', dest='project', action="store_true")
    parser.add_argument('--live', default='l')
    parser.add_argument('--pretrained', dest='pretrained', action="store_true")
    parser.add_argument('--ClipCaptionModel', dest='ccm', action="store_true")
    parser.add_argument('--prefix_length', type=int, default=10)
    parser.add_argument('--clip_length', type=int, default=10)
    parser.add_argument('--prefix_size', type=int, default=512)
    parser.add_argument('--coco', dest='coco', action="store_true")
    parser.add_argument('--conceptual', dest='conceptual', action="store_true")
    parser.add_argument('--transformer', dest='transformer', action="store_true")

    args = parser.parse_args()
    
    model = load_checkpoint(args)
    # caption_from_device(args, model)
    caption_live(model, args)

if __name__ == '__main__':
   main()
