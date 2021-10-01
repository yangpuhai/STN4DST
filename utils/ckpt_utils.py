import wget
import os
import torch

def download_ckpt(ckpt_path, target_path='bert-base-uncased'):
    url_path = "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin"
    print('start download %s from huggingface' % 'bert-base-uncased')
    wget.download(url_path, out=os.path.join(target_path, 'pytorch_model.bin'))
    return ckpt_path
    