import os
import copy
import argparse
from collections import deque

import torch
import torch.optim as opt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from rich import print
from datasets import load_dataset
from sacrebleu.metrics import BLEU

from transformers import T5TokenizerFast, T5ForConditionalGeneration
from utils import CommonGenEval, get_lm_input
from main import get_outputs, write_output
from slogging import Logger


def train_step(model, encoding, targets):

    input_ids, attention_mask = encoding.input_ids.to(
        device
    ), encoding.attention_mask.to(device)

    with torch.no_grad():
        actions = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=32,
            do_sample=True,
        )
        states = get_lm_input(
            model, input_ids=input_ids, attention_mask=attention_mask, labels=actions
        )

        actions = actions[:, 1:]
        states = states[:, :-1, :]

        bleu = BLEU()
        rewards = torch.zeros(states.shape[0], actions.shape[1])

        for j in range(actions.shape[1]):
            generated = tokenizer.batch_decode(
                actions[:, : j + 1], skip_special_tokens=True
            )
            for i in range(len(generated)):
                rewards[i, j] = bleu.corpus_score([generated[i]], targets[i]).score

    mean_reward = torch.mean(rewards[:, -1])

    for i in reversed(range(1, rewards.shape[1])):
        rewards[:, i] = rewards[:, i] - rewards[:, i - 1]

    gamma = 0.99
    for i in reversed(range(rewards.shape[1] - 1)):
        rewards[:, i] = rewards[:, i] + gamma * rewards[:, i + 1]

    rewards = rewards.to(device)
    mask = actions != 0

    states = states.reshape(-1, states.shape[-1])
    actions = actions.flatten()
    rewards = rewards.flatten()
    mask = mask.flatten()

    states = states[mask]
    actions = actions[mask]
    rewards = rewards[mask]

    return states, actions, rewards, mean_reward


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("model_type")
    parser.add_argument("model_weights")
    parser.add_argument("--batch_size", type=int, default=512)

    args = parser.parse_args()

    os.makedirs(f"./logs/{args.model_type}-rl/", exist_ok=True)
    Logger.init(f"./logs/{args.model_type}-rl/", "test.txt", True, log_freq=10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = T5TokenizerFast.from_pretrained(args.model_type)
    full = T5ForConditionalGeneration.from_pretrained(args.model_type).to(device)
    full.load_state_dict(torch.load(args.model_weights))

    model = copy.deepcopy(full.lm_head)

    if args.model_type == "t5-small":
        input_dim = 512
    elif args.model_type == "t5-base":
        input_dim = 768
    elif args.model_type == "t5-large":
        input_dim = 1024

    value_model = nn.Sequential(
        nn.Linear(input_dim, 512), nn.ReLU(), nn.Linear(512, 1)
    ).to(device)

    opt = opt.Adam(list(model.parameters()) + list(value_model.parameters()), lr=1e-2)

    dataset = load_dataset("common_gen")

    def collate_fn(batch):
        concepts = [" ".join(b["concepts"]) for b in batch]
        targets = [b["target"] for b in batch]
        concepts = tokenizer(concepts, padding=True, return_tensors="pt")
        targets = tokenizer(targets, padding=True, return_tensors="pt")
        return concepts, targets

    train_dl = DataLoader(
        CommonGenEval("train", tokenizer),
        batch_size=args.batch_size,
        drop_last=True,
        shuffle=True,
        collate_fn=CommonGenEval.collate_fn,
    )
    test_dl = DataLoader(
        dataset["validation"],
        batch_size=64,
        drop_last=False,
        shuffle=False,
        collate_fn=collate_fn,
    )

    N = 100
    steps = 0

    rew = deque(maxlen=N)
    policy_losses = deque(maxlen=N)
    value_losses = deque(maxlen=N)

    best_rew = -1

    for epoch in range(10):
        print(f"Epoch: {epoch}")
        for encoding, targets in train_dl:

            full.lm_head = copy.deepcopy(model)

            if steps % 25 == 0:
                print("Saving...")
                states, generated, examples = get_outputs(
                    full, device, tokenizer, test_dl, num_beams=5
                )
                write_output(f"rl_{steps}", states, generated, examples)
                torch.save(
                    model.state_dict(), f"{Logger.output_dir}/rl_head_{steps}.pth"
                )

            states, actions, rewards, mr = train_step(full, encoding, targets)
            rew.append(mr)

            opt.zero_grad()

            logits = model(states)
            logits = F.log_softmax(logits, dim=-1)
            logits = torch.gather(logits, 1, actions.unsqueeze(1)).squeeze(0)

            values = value_model(states).flatten()
            value_loss = F.mse_loss(values, rewards)

            policy_loss = -1 * torch.mean(
                (rewards.view(-1, 1) - values.detach().view(-1, 1)) * logits
            )

            policy_loss.backward()
            value_loss.backward()

            value_losses.append(value_loss.item())
            policy_losses.append(policy_loss.item())

            print(steps)
            print(np.mean(rew))
            print(np.mean(value_losses))
            print(np.mean(policy_losses))
            print()

            if np.mean(rew) > best_rew and len(rew) == N:
                best_rew = np.mean(rew)
                torch.save(model.state_dict(), f"{Logger.output_dir}/rl_head_best.pth")

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            steps += 1
