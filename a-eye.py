import glob
import cv2
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
from train import ClipCaptionModel, ClipCaptionPrefix
from collections import OrderedDict
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights_root = './conceptual_train'
checkpoint = ''
pretrained_model = ''
w_c=[]
list_models = glob.glob(f"{weights_root}/*")
latest_model = max(list_models, key=os.path.getctime)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load(os.path.join(latest_model), map_location=device)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
prefix_length = 10
model = ClipCaptionModel(prefix_length, clip_length=10, prefix_size=512,
                                num_layers=8, mapping_type='transformer')
adjusted_checkpoint = OrderedDict()
for k, v in checkpoint.items():
    name = k.replace('clip_project', 'clip_model') # remove `module.`
    adjusted_checkpoint[name] = v
model.load_state_dict(adjusted_checkpoint)
clip_model, preproce = clip.load('ViT-B/32', device=device)
model.eval()
clip_model.eval()
model.to(device=device)
clip_model.to(device=device)


def generate_caption(PIL_image ): 
    start_time = time.time() 
    with torch.no_grad():
        image = preproce(PIL_image).unsqueeze(0).to(device)
        prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)
        prefix = prefix / prefix.norm(2, -1).item()
        prefix_embed = model.clip_model(prefix).reshape(1, prefix_length, -1)
        captopn = generate2(model, tokenizer, embed=prefix_embed)
    print("--- %s seconds to load model ---" % (time.time() - start_time))
    return captopn   

def read_caption(caption):
    with torch.no_grad():
        start_time = time.time()
        stn_tst = get_text(caption, hps)
        x_tst = stn_tst.unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
        audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.float().numpy()
        sd.play(audio, blocking=False, samplerate=hps.data.sampling_rate)
        
        print("--- %s seconds to read the generated captions ---" % (time.time() - start_time))
        # wv.read("generated_voice.wav", audio ,rate=hps.data.sampling_rate, sampwidth=1)
        # wav_obj = sa.WaveObject.from_wave_file("generated_voice.wav")
        # wav_obj.play()).wait_done()

def caption ():
    test_data_path = os.path.join(os.getcwd(),'data/conceptual/test')
    start_time = time.time()
    images_to_generate_caption = []
    for file in os.listdir(test_data_path):
        image_path = os.path.join(os.path.join(os.getcwd(), test_data_path), file)
        image = Image.open(image_path)
        images_to_generate_caption.append(image)
      
    for image in images_to_generate_caption:
        image_cv = np.array(image)
        cv.imshow('test', image_cv)
        keypress = cv.waitKey(1)
        caption = generate_caption(image)
        read_caption(caption)

    print("--- %s overal time ---" % (time.time() - start_time))

def screen():
    cap = cv2.VideoCapture(0)
    ret,frame = cap.read()
    frame = PIL.Image.fromarray(frame)
    return frame

if __name__ == '__main__':
    image = screen()
    capt = generate_caption(image)
    read_caption(capt)
