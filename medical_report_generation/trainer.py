from __future__ import annotations

import time
from pathlib import Path
from typing import cast

import torch
from torch import nn
from torch.utils.data import DataLoader

from .datasets import IUXrayDataset
from .model import MedicalReportGeneration
from .utils import State

ckpt_dir = Path("ckpt")
ckpt_dir.mkdir(exist_ok=True)


class Trainer:
    def __init__(
        self,
        device: str = "cpu",
        finetune: bool = True,
        pth_file: str | None = None,
        dataset: IUXrayDataset | None = None,
    ) -> None:
        if finetune:
            learning_rate = 5e-5
        else:
            learning_rate = 1e-3
        self.model = MedicalReportGeneration(finetune=finetune, device=device).train()
        self.device = torch.device(device)
        self.model.to(self.device)
        if dataset is None: 
            dataset = IUXrayDataset()
        print(f"Loaded {len(dataset)} samples")
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), learning_rate)
        if pth_file is None:
            state = State(
                epoch=0,
                model_state_dict=self.model.state_dict(),
                optim_state_dict=self.optimizer.state_dict(),
                loss=0,
                all_losses=[],
            )
        else:
            state = cast(State, torch.load(pth_file, self.device))
            self.model.load_state_dict(state["model_state_dict"])
            self.optimizer.load_state_dict(state["optim_state_dict"])
            # get parameters view
            state["model_state_dict"] = self.model.state_dict()
            state["optim_state_dict"] = self.optimizer.state_dict()
        self.state = state

    def train_one_epoch(self, create_checkpoint: bool = True):
        loss_history = []
        for idx, (images, labels) in enumerate(self.dataloader):
            images = images.to(self.device)
            # labels = labels.to(self.device)
            self.optimizer.zero_grad()
            loss = self.model(images, labels)
            loss.backward()
            loss_history.append(loss.item())
            self.optimizer.step()
            print("iter: {}  loss: {:.4f}".format(idx, loss))
        self.state["epoch"] += 1
        loss_avg = sum(loss_history) / len(loss_history)
        self.state["loss"] = loss_avg
        self.state["all_losses"].append(loss_avg)
        print(f"epoch {self.state['epoch']} training complete")
        print(f"STAT LOSS: {self.state['loss']:.4f}")
        if create_checkpoint:
            checkpoint = ckpt_dir / f"model_{getattr(self.model, "version", "v1")}_epoch{self.state['epoch']}.pth"
            torch.save(self.state, checkpoint)
            print(f"model saved at {checkpoint}")

    def train(self, num_epochs: int = 100):
        for _ in range(num_epochs):
            self.train_one_epoch(True)
