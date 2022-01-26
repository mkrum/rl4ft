import os
import time
import torch
from collections import deque
import numpy as np


class Logger:

    output_dir: str = None
    log_file: str = None
    overwrite: bool = False
    log_freq: int = 100
    checkpoint_freq: int = 10000

    _losses = None
    _epoch_num = 0

    @classmethod
    def init(cls, output_dir=".", log_file=".", overwrite=False, log_freq=100):
        cls.output_dir = output_dir
        cls.log_freq = log_freq
        log_path = f"{output_dir}/{log_file}"

        if not overwrite and os.path.exists(log_path):
            res = input(f"File {log_path} exists. Overwrite? (Y/n) ")
            go_ahead = res == "" or res.lower() == "y"

            if not go_ahead:
                exit()

        cls.log_file = open(log_path, "w")

    @classmethod
    def save_checkpoint(cls, model, opt):
        torch.save(model.state_dict(), f"{cls.output_dir}/model_{cls._epoch_num}.pth")
        torch.save(opt.state_dict(), f"{cls.output_dir}/opt.pth")

    @classmethod
    def test_output(cls, acc, loss):
        print(f"({cls._epoch_num}) Acc: {round(acc, 2)} Test Loss: {loss}")
        cls.log_file.write(f"TEST {cls._epoch_num} {time.time()} {acc}\n")

    @classmethod
    def epoch(cls):
        cls._epoch_num += 1
        cls._losses = deque(maxlen=1000)

    @classmethod
    def loss(cls, model, opt, batch_idx, total, loss):
        cls._losses.append(loss)

        cls.log_file.write(f"TRAIN {cls._epoch_num} {batch_idx} {time.time()} {loss}\n")

        if batch_idx > 0 and batch_idx % cls.log_freq == 0:
            print(
                f"({cls._epoch_num} {batch_idx}/{total}) Loss (Avg): {np.mean(cls._losses)}"
            )

        if batch_idx > 0 and batch_idx % cls.checkpoint_freq == 0:
            cls.save_checkpoint(model, opt)
