import argparse
import pandas as pd
import warnings
warnings.filterwarnings('ignore') # turn off the warning messages

import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification


def inference(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir, num_labels=args.num_labels)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    softmax = torch.nn.Softmax(-1)
    dataset = pd.read_csv(args.data_dir)

    for i in range(len(dataset)):
        texts = dataset.loc[i]['text']
        tokens = tokenizer(texts, return_tensors='pt').to(device)
        tokens_len = len(tokens[0])

        if tokens_len < 512:
            with torch.no_grad():
                output = model(**tokens)
                output_flag = output.logits.argmax(-1).cpu().tolist()[0]
                output_value = softmax(output.logits).max(-1).values.cpu().tolist()[0]*100

            print(f'- {i+1}번 데이터: {texts}\n')
            print(f'- {i+1}번 예측: {output_value:.2f}%  (0: 정상 / 1: 공격)\n')
            print(f'----------------------------\n')

        else:
            print(f'- {i+1}번 데이터: {texts}\n')
            print(f'- {i+1}번 예측: 판단불가 (토큰길이 초과))\n')
            print(f'----------------------------\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='./dataset/dataset_prep2.csv')
    parser.add_argument("--model_dir", type=str, default='./ckpt/SecureBERT')
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--tokenizer_dir", type=str, default='./ckpt/SecureBERT')
    args = parser.parse_args()

    inference(args)