### config.py ###
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
