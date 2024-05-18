from medical_report_generation.trainer import Trainer

trainer = Trainer(device="cuda", pth_file="ckpt/model_v1_epoch6.pth")

trainer.train()
