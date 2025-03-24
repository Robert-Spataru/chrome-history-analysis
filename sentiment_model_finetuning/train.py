import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from dataclass import *
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from training_loop import *
from model import *

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set hyperparameters
PRETRAINED_MODEL_NAME = "siebert/sentiment-roberta-large-english"
BATCH_SIZE = 16
MAX_LENGTH = 128
EPOCHS = 5
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
RANDOM_SEED = 42

# Set seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# Extract data from synthetic_data.csv
filepath = "synthetic_training_data/synthetic_data.csv"
webpage_titles, sentiment_scores = get_data_from_filepath(filepath)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

# Load datasets
dataset = WebpageTitleSentiment(webpage_titles, sentiment_scores, tokenizer)

# Split data
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Initialize model
model1 = SentimentRegressionModelWithLoRA(PRETRAINED_MODEL_NAME).to(device)
model2 = SentimentRegressionModel(PRETRAINED_MODEL_NAME).to(device)

# Print model architecture summary
print(model1)
print(model1)

model_list_dict = {"LoRA_Finetuned":model1, "Last_Transormer_Layer_Finetuned":model2}

for model_name, model in model_list_dict.items():
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params} ({trainable_params/all_params:.2%} of all parameters)")

    # Train model
    trained_model = train_sentiment_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # Save final model
    torch.save(trained_model.state_dict(), f"{model_name}.pt")

