from medical_report_generation.trainer import Trainer

trainer = Trainer(device="cuda", finetune=False, local_files_only=True)

trainer.train(50)
