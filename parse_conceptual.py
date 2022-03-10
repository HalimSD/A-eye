import torch
import clip
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pickle
from tqdm import tqdm
import os
import csv
import threading
import requests
import shutil
import PIL
from typing import List, Tuple, Optional
import argparse
from pathlib import Path

class ConceptualDS(Dataset):

    @staticmethod
    def get_all_data(data_root: str, suffix: str):
        data = []
        for i in range(2):
            out_data_path = f"{data_root}/conceptual_{suffix}_{i:02d}.pkl"
            if os.path.isfile(out_data_path):
                with open(out_data_path, 'rb') as f:
                    raw_data = pickle.load(f)["info"]
                data.append(raw_data)

        return data

    @staticmethod
    def collect(data_root: str, suffix: str):
        raw_data = ConceptualDS.get_all_data(data_root, suffix)
        data = []
        for thread_data in raw_data:
            for item in thread_data:
                data.append((item, thread_data[item]["caption"]))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        image_name, caption = self.data[item]
        image_path = f"{self.data_root}/{self.suffix}/{image_name}.jpg"
        is_error = False
        image = self.dummy
        try:
            image = Image.open(image_path) #.resize(224)
            image = self.preprocess(image)
        except PIL.UnidentifiedImageError:
            is_error = True
        except OSError:
            is_error = True
        except BaseException:
            is_error = True
        if is_error:
            return image, "", image_name
        return image, caption, image_name

    def __init__(self, data_root: str, preprocess, suffix: str):
        self.suffix = suffix
        self.data_root = data_root
        self.data = self.collect(data_root, suffix)
        self.preprocess = preprocess
        self.dummy = torch.zeros(3, 224, 224)


def save_pickle(data, out_path: str, recover_index: Optional[int] = None):
    if os.path.isfile(out_path) and recover_index is not None:
        recover_path = f'{out_path[:-4]}_{recover_index:02d}.pkl'
        shutil.copyfile(out_path, recover_path)
    with open(out_path, 'wb') as f:
        pickle.dump(data, f)


def get_image(url: str, out_path: str, timeout=10):
    try:
        r = requests.get(url, stream=True, timeout=timeout)
        if r.status_code == 200:
            with open(out_path, 'wb') as f:
                r.raw.decode_content = True
                shutil.copyfileobj(r.raw, f)
            return True
        return False
    except BaseException:
        return False


def thread(urls: List[Tuple[List[str], int]], thread_id: int, progress: tqdm, lock: Optional[threading.Lock],
           suffix: str, conceptual_root: str):
    out_root = f"{conceptual_root}/{suffix}"
    out_data_path = f"{conceptual_root}/conceptual_{suffix}_{thread_id:02d}.pkl"
    recover_index = 0
    if os.path.isfile(out_data_path):
        with open(out_data_path, 'rb') as f:
            data = pickle.load(f)
        parsed = data['parsed']
        info = data['info']
    else:
        parsed = set()
        info = {}
    for i in range(0, len(urls)):
        (caption, url), ind = urls[i]
        name = f"{ind:08d}"
        out_path = f"{out_root}/{name}.jpg"
        if url not in parsed and not os.path.isfile(out_path) and get_image(url, out_path):
            parsed.add(url)
            info[name] = {"url": url, "caption": caption}
        if lock is not None:
            lock.acquire()
            try:
                progress.update()
            finally:
                lock.release()
        else:
            progress.update()
        if (i + 1) % 1000 == 0:
            save_pickle({'parsed': parsed, 'info': info}, out_data_path, recover_index)
            recover_index = 1 - recover_index
    save_pickle({'parsed': parsed, 'info': info}, out_data_path, 2)
    return 0


def download_conceptual(conceptual_root: str, num_threads: int, num_images:int):
   
    urls = []
    for suffix in ("train","val"):
        if suffix == "train":
            training_path = os.path.join(conceptual_root, 'Train_GCC-training.tsv')
            with open(training_path) as f:
                lines = f.readlines()
                lines = lines[:num_images]
            sub_set_path = '%s/subset_Train_GCC-training.tsv' %(conceptual_root)
            if not os.path.exists(sub_set_path):
                myfile = Path(sub_set_path)
                myfile.touch(exist_ok=True)
            with open(sub_set_path, 'w') as f:
                for line in lines:
                    f.write(line) 

            tsv_path = f"{conceptual_root}/subset_Train_GCC-training.tsv"
        else:
            tsv_path = f"{conceptual_root}/Validation_GCC-1.1.0-Validation.tsv"
        with open(tsv_path) as f:
            read_tsv = csv.reader(f, delimiter="\t")
            for i, row in enumerate(read_tsv):
                urls.append((row, i))
        progress = tqdm(total=len(urls))
        if num_threads == 1:
            thread(urls, 0, progress, None, suffix, conceptual_root)
        else:
            groups = []
            threads = []
            lock = threading.Lock()
            split_size = len(urls) // num_threads
            for i in range(num_threads):
                if i < num_threads - 1:
                    groups.append(urls[i * split_size: (i + 1) * split_size])
                else:
                    groups.append(urls[i * split_size:])
            for i in range(num_threads):
                threads.append(threading.Thread(target=thread, args=(groups[i], i, progress, lock, suffix, conceptual_root)))
            for i in range(num_threads):
                threads[i].start()
            for i in range(num_threads):
                threads[i].join()
        progress.close()


def add_period(caption: str):
    caption = caption.strip()
    if caption[-1] != '.':
        caption = caption + '.'
    elif caption[-2] == ' ':
        caption = caption[:-2] + '.'
    return caption


def create_clip_embeddings(conceptual_root: str, clip_model_type: str):
    all_embeddings = []
    all_captions = []
    for suffix in ("train", "val"):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
        # Load proj model
        print('Load proj model')
        # model = ClipCaptionPrefix(prefix_length, clip_length=clip_length, \
        #                                 prefix_size=prefix_size, num_layers=8, mapping_type='transformer')
        clip_model = clip_model.eval()
        ds = ConceptualDS(conceptual_root, preprocess, suffix)
        dl = DataLoader(ds, batch_size=20, shuffle=False, drop_last=True)
        progress = tqdm(total=len(dl))
        counter = 0
        clip_model_name = clip_model_type.replace('/', '_')
        out_data_path = f"{os.path.join(os.getcwd(), 'checkpoints/data_parsed')}/conceptual_{clip_model_name}_{suffix}.pkl"
        recover_index = 0
        for i, data in enumerate(dl):
            images, captions, image_names = data
            images = images.to(device)
            with torch.no_grad():
                prefix = clip_model.encode_image(images).to(device)
            is_valid = list(map(lambda x: x != "", captions))
            mask = torch.tensor(is_valid)
            all_embeddings.append(prefix[mask])
            captions = [caption for j, caption in enumerate(captions) if is_valid[j]]
            image_names = [image_name for j, image_name in enumerate(image_names) if is_valid[j]]
            all_captions.extend([{"caption": add_period(caption), "clip_embedding": counter + j, "image_id": image_name}
                                 for j, (caption, image_name) in enumerate(zip(captions, image_names))])
            progress.update()
            counter += len(captions)
            if (i + 1) % 1000 == 0:
                save_pickle({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, out_data_path, recover_index)
                recover_index = 1 - recover_index
        save_pickle({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, out_data_path, 2)
        progress.close()

    return 0
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='./data/conceptual')
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--num_images', type=int, default=100)
    args = parser.parse_args()
    # download_conceptual(args.data_root, args.num_threads, args.num_images)
    create_clip_embeddings(args.data_root, args.clip_model_type)

if __name__ == '__main__':
    main()
