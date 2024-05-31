from medical_report_generation.trainer import Trainer

trainer = Trainer(device="cuda", finetune=False)

trainer.train(50)
