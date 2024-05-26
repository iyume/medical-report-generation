import torch
from PIL import Image

from medical_report_generation.model import MedicalReportGeneration
from medical_report_generation.utils import State

state: State = torch.load("ckpt/model_v1_epoch69.pth", map_location="cpu")

model = MedicalReportGeneration("cpu")
model.load_state_dict(state["model_state_dict"])

text = model.generate(Image.open("iu_xray/images/CXR3935_IM-2006/0.png"))
print(text)
