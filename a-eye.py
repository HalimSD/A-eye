import torch
import clip
import cv2 as cv
import wavio as wv
from utils.clip_synthesized import generate2, hps, net_g, get_text, model, prefix_length
from transformers import GPT2Tokenizer
import PIL 
from skimage import io
import simpleaudio as sa

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def caption ():
    cam = cv.VideoCapture(0)

    while True:
        check, frame = cam.read()
        cv.imshow('video', frame)
        keypress = cv.waitKey(1)
        
        if keypress & 0xFF == ord('q'):
            break
        if keypress & 0xFF == ord('c'):
            cv.imwrite('example.png',frame)
            pil_image = PIL.Image.fromarray(io.imread('example.png'))
            image = preprocess(pil_image).unsqueeze(0).to(device)
            with torch.no_grad():
                prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)
                prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)
                caption = generate2(model, tokenizer, embed=prefix_embed)
                stn_tst = get_text(caption, hps)
                x_tst = stn_tst.unsqueeze(0)
                x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
                audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.float().numpy()
            wv.write("generated_voice.wav", audio ,rate=hps.data.sampling_rate, sampwidth=1)
            wav_obj = sa.WaveObject.from_wave_file("generated_voice.wav")
            play_obj = wav_obj.play()
            play_obj.wait_done()

    cam.release()
    cv.destroyAllWindows()

caption()