from medical_report_generation.trainer import Trainer

trainer = Trainer(device="cuda")

trainer.train(50)
