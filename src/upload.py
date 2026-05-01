from nets.DiT import DIT

model = DIT.load_from_checkpoint("./model_ckpts/epoch=0-step=700.ckpt")
model.push_to_hub("Jumpr/artbench-dit-700-steps")