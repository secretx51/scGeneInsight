class HyperParameters():
    # Hyperparameters
    # TODO: Tune, specifically try single layer hidden dim
    NUM_GENES = 1000
    TEST_SPLIT = 0.80
    BATCH_SIZE = 96 # set 96
    HIDDEN_DIM = [1024, 512]
    DROPOUT_RATE = 0.3
    DROP_CONNECT = 0.2
    L1_REG = 0.005
    LR = 0.001
    WEIGHT_DECAY = 0
    NUM_EPOCHS = 1000
    NOISE_AMOUNT = 0.4
    MOMENTUM = 0.9
    QUANTILES = [0.01, 0.05, 0.1, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]