import os
from pathlib import Path
import cv2
import torch
import time 
import clip
import argparse
import wavio as wv
import PIL
from PIL import Image
#import sounddevice as sd
import simpleaudio as sa
from transformers import GPT2Tokenizer
from utils.clip_synthesized import generate2, hps, net_g, get_text
from models.clip_caption_model import ClipCaptionModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
clip_model, preprocess = clip.load('ViT-B/32', device=device)

os.path.join(os.path.join(os.getcwd(),'data'), 'test')

WEIGHTS_PATHS = {
'project_conceptual': Path('data/project_weights/400_imgs'),
'pretrained_conceptual': Path('data/pretrained_weights')
}

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
        test_data_path = Path('data/test')

        image_paths = [os.path.join(test_data_path, name) for name in os.listdir(test_data_path) if name[-4] == '.']
        img_list = [Image.open(image) for image in image_paths]    
        for image in img_list:
                caption = generate_caption(image, model )
        print("--- %s overal time ---" % (time.time() - start_time))
       
def screen():
    cap = cv2.VideoCapture(0)
    ret,frame = cap.read()
    frame = PIL.Image.fromarray(frame)
    return frame

def caption_live(model):
    cam = cv2.VideoCapture(0)
    while True:
        x, frame = cam.read()
        cv2.imshow('video', frame)
        keypress = cv2.waitKey(1000)
        if keypress & 0xFF != ord('q'):
            pil_image = PIL.Image.fromarray(frame)
            caption = generate_caption(pil_image, model)
            read_caption(caption)
        elif keypress & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()

def last_model (args: argparse.Namespace):
    if args.project and args.conceptual:
        model_path = WEIGHTS_PATHS.get('project_conceptual')
        list_models_path = os.listdir(model_path)
        weights_list = []
        for weight_path in list_models_path:
            if '.pt' in weight_path:
                path = os.path.join(model_path, weight_path)
                weights_list.append(path) 
        latest_model = max(weights_list, key=os.path.getctime)
        return latest_model
    elif args.pretrained and args.conceptual:
        model_path = WEIGHTS_PATHS.get('pretrained_conceptual')
        return model_path 
    else:
        assert 'Arguments are not complete'
    

def load_checkpoint(args: argparse.Namespace):
    latest_model = last_model(args)
  
    if args.project and args.conceptual:
        print('LOADING: Conceptual Project')
        checkpoint = torch.load(latest_model , map_location='cpu')
        model = ClipCaptionModel(args.prefix_length, args.prefix_size)
        model.load_state_dict(checkpoint)
        model.eval()
        model.to(device=device)
        return model  

    elif args.pretrained and args.conceptual:
        print('LOADING: Conceptual Pretrained')
        checkpoint = torch.load(latest_model, map_location=device)
        model = ClipCaptionModel(prefix_length = 10, prefix_size = 512)
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
    print(type(model))
    # caption_live(model) 
    caption_from_device(model)
    
if __name__ == '__main__':
    main()
