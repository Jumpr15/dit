from nets.DiT import DIT

# set model checkpoint
model = DIT.load_from_checkpoint("./model_ckpts/")

#
model.push_to_hub("Jumpr/anime-dit-checkpoints")