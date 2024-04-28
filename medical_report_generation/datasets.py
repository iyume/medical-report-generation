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
        if type == "train":
            self.annotations = all_annotations["train"] + all_annotations["val"]
        else:
            self.annotations = all_annotations["test"]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        annotation = self.annotations[index]
        pil_images = [
            Image.open(os.path.join(self.iu_xray_path, "images", i)).convert("RGB")
            for i in annotation["image_path"]
        ]
        return _transform(pil_images[0]), annotation["report"]

    def __len__(self):
        return len(self.annotations)
