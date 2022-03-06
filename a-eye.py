from utils.clip_synthesized import generate2, hps, net_g, get_text, ClipCaptionModel
import os
import torch
import clip
import cv2 as cv
import wavio as wv
from transformers import GPT2Tokenizer
from PIL import Image 
import simpleaudio as sa
import numpy as np
import time 


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights_path = "./pretrained_models/conceptual_weights.pt"

def caption ():
    test_data_path = os.path.join(os.getcwd(),'data/conceptual/test')
    for file in os.listdir(test_data_path): 
        start_time = time.time()
        image_path = os.path.join(os.path.join(os.getcwd(), test_data_path), file)
        image = Image.open(image_path)
        pil_image_Array = np.array(image)
        cv.imshow('test', pil_image_Array)
        keypress = cv.waitKey(1)

        with torch.no_grad():
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            clip_model, preprocess = clip.load(
                "ViT-B/32", device=device, jit=False
            )
            image = preprocess(image).unsqueeze(0).to(device)
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            prefix_length = 10
            model = ClipCaptionModel(prefix_length)
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict)
            model = model.eval()
            model = model.to(device)
            prefix = clip_model.encode_image(image).to(
                device, dtype=torch.float32
            )
            prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)
            caption = generate2(model, tokenizer, embed=prefix_embed)
            print("--- %s seconds to read and generate captions ---" % (time.time() - start_time))
            stn_tst = get_text(caption, hps)
            x_tst = stn_tst.unsqueeze(0)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
            audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.float().numpy()
            wv.write("generated_voice.wav", audio ,rate=hps.data.sampling_rate, sampwidth=1)
            wav_obj = sa.WaveObject.from_wave_file("generated_voice.wav")
            play_obj = wav_obj.play()
            play_obj.wait_done()
            if keypress & 0xFF == ord('q'):
                cv.destroyAllWindows()
caption()