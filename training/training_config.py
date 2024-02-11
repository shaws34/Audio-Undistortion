# training_config.py

# General Training Configuration
config = {
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 20,
    "dataset_path": "training/data/",
    "model_save_path": "training/saved_models/",
    "log_interval": 10,  # Log training progress every N batches
    "val_interval": 1,   # Validate model every N epochs
    "device": "cuda" if torch.cuda.is_available() else "cpu",  # Automatically use GPU if available
}

# Neural Network Architecture Specific Configuration
# This can vary widely depending on your specific use case and the model architecture you choose.
# For example, if you're using a simple CNN for image classification, you might specify:
nn_config = {
    "input_size": 784,  # For MNIST, 28x28 images
    "hidden_layers": [128, 64],  # Sizes of hidden layers
    "output_size": 10,  # For MNIST, 10 classes
    "activation_function": "ReLU",
}

# Training Data Augmentation/Preprocessing Configuration
# Specify any data augmentation or preprocessing parameters. For instance:
data_augmentation_config = {
    "rotation_range": 10,  # degrees
    "scale_range": 0.1,    # Scale images by +/- 10%
    "shift_range": 0.1,    # Shift images by +/- 10% of height/width
    # Add more augmentation parameters as needed
}

# Optimizer Configuration
# While a learning rate is specified in the general config, other optimizer-specific parameters can be defined here.
optimizer_config = {
    "momentum": 0.9,
    "weight_decay": 1e-4,
    # Add more optimizer-specific parameters as needed
}
