import json
import os
from pathlib import Path
from typing import Callable, Literal

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

_transform: Callable[[Image.Image], torch.Tensor] = transforms.Compose([transforms.ToTensor()])


class IUXrayDataset(Dataset):
    def __init__(
        self,
        iu_xray_path: str = "iu_xray",
        type: Literal["train", "test"] = "train",
    ) -> None:
        self.iu_xray_path = iu_xray_path
        all_annotations = json.load(open(Path(iu_xray_path) / "annotation.json"))
        # train 2069, val 296, test 590
        if type == "train":
            self.annotations = all_annotations["train"] + all_annotations["val"]
        else:
            self.annotations = all_annotations["test"]
        self.pairs: list[tuple[str, str]] = []
        for ann in self.annotations:
            for image in ann["image_path"]:
                self.pairs.append((image, ann["report"]))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        image_path, report = self.pairs[index]
        image = Image.open(os.path.join(self.iu_xray_path, "images", image_path)).convert("RGB")
        return _transform(image), report

    def __len__(self):
        return len(self.pairs)
