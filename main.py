import os
import argparse
from typing import Any
import multiprocessing as mp
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torch.optim as opt
from torch.utils.data import DataLoader
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from datasets import load_dataset
from jury import Jury
from rich import print

from utils import get_outputs, write_output, make_collate
from slogging import Logger


def compute_loss(model, encoding, target_encoding):
    """
    Base loss computation via hugging face API
    """
    input_ids, attention_mask = encoding.input_ids.to(
        device
    ), encoding.attention_mask.to(device)
    labels = target_encoding.input_ids.to(device)
    return model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss


@dataclass(frozen=True)
class TrainingContext:

    eval_fn: Any
    train_fn: Any
    train_dl: Any
    test_dl: Any
    epochs: int = 21
    eval_freq: int = 10000

    def __call__(self, model, tokenizer, opt, device):

        for epoch in range(self.epochs):
            Logger.epoch()
            out = self.eval_fn(epoch, model, tokenizer, self.test_dl)
            Logger.test_output(*out)

            Logger.save_checkpoint(model, opt)

            for (batch_idx, (encoding, target_encoding)) in enumerate(self.train_dl):
                opt.zero_grad()

                loss = self.train_fn(model, encoding, target_encoding)

                loss.backward()
                opt.step()

                Logger.loss(model, opt, batch_idx, len(self.train_dl), loss.item())

        out = self.eval_fn(model, self.test_dl)
        Logger.test_output(*out)
        Logger.save_checkpoint(model, opt)


def run_eval(epoch, state_path, generated_path, examples_path):
    """
    Loads data, uses jury to compute the main metrics
    """

    states = open(state_path, "r").readlines()
    generated = open(generated_path, "r").readlines()
    examples = open(examples_path, "r").readlines()

    jury = Jury()

    uniq_states = list(set(states))

    grouped_generated = {u: [] for u in uniq_states}
    grouped_examples = {u: [] for u in uniq_states}
    for (i, s) in enumerate(states):
        grouped_generated[s].append(generated[i])
        grouped_examples[s].append(examples[i])

    generated = [grouped_generated[u] for u in uniq_states]
    examples = [grouped_examples[u] for u in uniq_states]
    scorer = Jury()
    out = scorer(predictions=generated, references=examples)
    out["epoch"] = epoch
    print(out)


def base_eval_fn(epoch, model, tokenizer, test_dl):
    """
    Writes output, runs evaluation in a seperate process
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    states, generated, examples = get_outputs(
        model, device, tokenizer, test_dl, num_beams=5
    )
    output_dir = f"{Logger.output_dir}/samples"
    files = write_output(output_dir, epoch, states, generated, examples)

    p = mp.Process(target=run_eval, args=(epoch, *files))
    p.daemon = True
    p.start()

    return 0.0, 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("model_type")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    log_dir = f"./logs/{args.model_type}"
    os.makedirs(log_dir, exist_ok=True)
    Logger.init(log_dir, "test.txt", True, log_freq=10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = T5TokenizerFast.from_pretrained(args.model_type)
    model = T5ForConditionalGeneration.from_pretrained(args.model_type).to(device)

    opt = opt.Adam(model.parameters(), lr=5e-5)

    dataset = load_dataset("common_gen")

    train_dl = DataLoader(
        dataset["train"],
        batch_size=args.batch_size,
        drop_last=True,
        shuffle=True,
        collate_fn=make_collate(tokenizer),
    )
    test_dl = DataLoader(
        dataset["validation"],
        batch_size=args.batch_size,
        drop_last=False,
        shuffle=False,
        collate_fn=make_collate(tokenizer),
    )

    ctx = TrainingContext(base_eval_fn, compute_loss, train_dl, test_dl)
    ctx(model, tokenizer, opt, device)
