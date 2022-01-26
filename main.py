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


@dataclass
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


def get_outputs(model, device, tokenizer, test_dl, do_sample=False, num_beams=None):
    """
    Runs inference, collects results into a list
    """
    states = []
    generated = []
    examples = []

    for (batch_idx, (encoding, targets)) in enumerate(test_dl):
        input_ids, attention_mask = encoding.input_ids.to(
            device
        ), encoding.attention_mask.to(device)
        labels = targets.input_ids.to(device)

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=32,
            do_sample=do_sample,
            num_beams=num_beams,
        )

        states += tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        generated += tokenizer.batch_decode(outputs, skip_special_tokens=True)
        examples += tokenizer.batch_decode(labels, skip_special_tokens=True)

    return states, generated, examples


def write_output(epoch, states, generated, examples):
    """
    Write the model output into text files for evaluation.
    """

    for i in range(10):
        print(f"[red]{states[i]} [green]-> [blue]{generated[i]}")

    output_dir = f"{Logger.output_dir}/samples"

    os.makedirs(output_dir, exist_ok=True)

    state_file = f"{output_dir}/states_{epoch}.txt"
    generated_file = f"{output_dir}/generated_{epoch}.txt"
    examples_file = f"{output_dir}/examples_{epoch}.txt"

    with open(state_file, "w") as f:
        for i in range(len(generated)):
            f.write(states[i] + "\n")

    with open(generated_file, "w") as f:
        for i in range(len(generated)):
            f.write(generated[i] + "\n")

    with open(examples_file, "w") as f:
        for i in range(len(generated)):
            f.write(examples[i] + "\n")

    return (state_file, generated_file, examples_file)


def base_eval_fn(epoch, model, tokenizer, test_dl):
    """
    Writes output, runs evaluation in a seperate process
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    states, generated, examples = get_outputs(
        model, device, tokenizer, test_dl, num_beams=5
    )
    files = write_output(epoch, states, generated, examples)

    p = mp.Process(target=run_eval, args=(epoch, *files))
    p.daemon = True
    p.start()

    return 0.0, 0.0


if __name__ == "__main__":
    # batch_size=256, #t5-small
    # batch_size=155, #t5-base
    # batch_size=16, #t5-large

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

    def collate_fn(batch):
        concepts = [" ".join(b["concepts"]) for b in batch]
        targets = [b["target"] for b in batch]
        concepts = tokenizer(concepts, padding=True, return_tensors="pt")
        targets = tokenizer(targets, padding=True, return_tensors="pt")
        return concepts, targets

    train_dl = DataLoader(
        dataset["train"],
        batch_size=args.batch_size,
        drop_last=True,
        shuffle=True,
        collate_fn=collate_fn,
    )
    test_dl = DataLoader(
        dataset["validation"],
        batch_size=args.batch_size,
        drop_last=False,
        shuffle=False,
        collate_fn=collate_fn,
    )

    ctx = TrainingContext(base_eval_fn, compute_loss, train_dl, test_dl)
    ctx(model, tokenizer, opt, device)
