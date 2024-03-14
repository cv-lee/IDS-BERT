import os
import argparse
import warnings
warnings.filterwarnings('ignore') # turn off the warning messages

import torch
import transformers

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig

from utils import load_txt, compute_metrics

def dataset_gen(dataset, batch_size=1000):
    return (
        dataset["train"][i : i + batch_size]['text']
        for i in range(0, dataset['train'].num_rows, batch_size)
    )


def train_tokenizer(tokenizer, dataset, special_tokens, vocab_size, save_dir=None):
    tokenizer = tokenizer.train_new_from_iterator(dataset_gen(dataset, batch_size=1000),
                                                  new_special_tokens=special_tokens,
                                                  vocab_size=vocab_size)

    if save_dir is not None:
        tokenizer.save_pretrained(save_dir)

    return tokenizer


def train(args):

    # Load model & tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(args.model_load_dir, num_labels=args.num_labels)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_load_dir)

    # Train tokenizer
    dataset = load_dataset('csv', data_files={'train': args.data_dir})

    if args.tokenizer_train:
        special_tokens = load_txt(args.special_tokens_dir)
        tokenizer = train_tokenizer(tokenizer = tokenizer,
                                    dataset = dataset,
                                    special_tokens = special_tokens,
                                    vocab_size = tokenizer.vocab_size,
                                    save_dir = args.tokenizer_save_dir)

    # Set dataset
    dataset = dataset.map(
        lambda example: tokenizer(example['text'], max_length=512, padding='max_length', truncation=True, return_tensors="pt"),
        batched=True
    )
    dataset = dataset.shuffle()

    # Train model
    trainer = transformers.Trainer(
        model = model,
        train_dataset = dataset["train"],
        eval_dataset = dataset["train"],
        compute_metrics = compute_metrics,
        args = transformers.TrainingArguments(
            output_dir = args.model_save_dir,
            learning_rate = args.learning_rate,
            per_device_train_batch_size = args.per_device_train_batch_size,
            per_device_eval_batch_size = args.per_device_eval_batch_size,
            gradient_accumulation_steps = args.gradient_accumulation_steps,
            num_train_epochs = args.num_train_epochs,
            do_eval = args.do_eval,
            evaluation_strategy = args.evaluation_strategy,
            eval_steps = args.eval_steps,
            weight_decay = args.weight_decay,
            warmup_steps = args.warmup_steps,
            fp16 = args.fp16,
            save_strategy = args.save_strategy,
            optim = args.optim,
            save_safetensors = args.save_safetensors,
            remove_unused_columns = args.remove_unused_columns
        )
    )
    model.config.use_cache = False
    trainer.train()

    #metrics = trainer.evaluate(eval_dataset=dataset["train"])
    #print(metrics)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default='./dataset/dataset_prep2.csv')

    parser.add_argument("--model_load_dir", type=str, default='./ckpt/pretrained/SecureBERT')
    parser.add_argument("--model_save_dir", type=str, default='./ckpt')
    parser.add_argument("--num_labels", type=int, default=2)

    parser.add_argument("--tokenizer_load_dir", type=str, default='./ckpt/pretrained/SecureBERT')
    parser.add_argument("--tokenizer_save_dir", type=str, default='./ckpt')
    parser.add_argument("--tokenizer_train", type=bool, default=False)
    parser.add_argument("--special_tokens_dir", type=str, default='./ckpt/manual_special_tokens.txt')

    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--do_eval", type=bool, default=True)
    parser.add_argument("--evaluation_strategy", type=str, default='steps')
    parser.add_argument("--eval_steps", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--fp16", type=bool, default=True)
    parser.add_argument("--save_strategy", type=str, default='epoch')
    parser.add_argument("--optim", type=str, default='paged_adamw_8bit')
    parser.add_argument("--save_safetensors", type=bool, default=True)
    parser.add_argument("--remove_unused_columns", type=bool, default=False)
    args = parser.parse_args()

    train(args)