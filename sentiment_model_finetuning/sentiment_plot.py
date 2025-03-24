import matplotlib.pyplot as plt
import os
from dataclass import *
from model import *
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch.nn.functional as F
import pickle


def compare_sentiment():
    device = torch.device("cpu")
    model_name = "siebert/sentiment-roberta-large-english"
    filepath = "synthetic_training_data/synthetic_data.csv"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    lora_finetuned_path = os.path.join(current_dir, "..", "model-parameters", "LoRA_Finetuned.pt")
    synthetic_data_path = os.path.join(current_dir, "..", "synthetic-training-data", "synthetic_data.csv")
    webpage_titles, sentiment_scores = get_data_from_filepath(synthetic_data_path)
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load datasets
    dataset = WebpageTitleSentiment(webpage_titles, sentiment_scores, tokenizer)
    model_finetuned = SentimentRegressionModelWithLoRA(model_name).to(device)
    model_finetuned.load_state_dict(torch.load(lora_finetuned_path, weights_only=True))
    model_finetuned.eval()  
    
    synthetic_data_labels = np.array(sentiment_scores)
    model_outputs = []
    with torch.no_grad():
        for i in range(len(webpage_titles)):
            tokenized_text = tokenizer(webpage_titles[i], return_tensors="pt").to(device)
            model_output = model_finetuned(**tokenized_text)
            model_outputs.append(model_output.detach().numpy())
    
    # Save using pickle
    with open('model_outputs.pkl', 'wb') as f:
        pickle.dump(model_outputs, f)
        
    with open('synthetic_data.pkl', 'wb') as f:
        pickle.dump(synthetic_data_labels, f)
    
    plt.plot(model_outputs, synthetic_data_labels, marker='o')
    plt.scatter(model_outputs, synthetic_data_labels)
    plt.title("Synthetic Data Labels v.s Fine Tuned Model Outputs")
    plt.xlabel("Fine Tuned Model Outputs")
    plt.ylabel("Synthetic Data Labels")
    plt.show()
    plt.savefig("line_plot.png")
    
    plt.scatter(model_outputs, synthetic_data_labels, marker='o')
    plt.title("Synthetic Data Labels v.s Fine Tuned Model Outputs")
    plt.xlabel("Fine Tuned Model Outputs")
    plt.ylabel("Synthetic Data Labels")
    plt.show()
    plt.savefig("scatter_plot.png")

if __name__ == "__main__":
    """
    device = torch.device("cpu")
    model_name = "siebert/sentiment-roberta-large-english"
    filepath = "synthetic_training_data/synthetic_data.csv"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    synthetic_data_path = os.path.join(current_dir, "..", "synthetic-training-data", "synthetic_data.csv")
    webpage_titles, sentiment_scores = get_data_from_filepath(synthetic_data_path)
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load datasets
    dataset = WebpageTitleSentiment(webpage_titles, sentiment_scores, tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()  
    
    synthetic_data_labels = np.array(sentiment_scores)
    model_outputs = []
    with torch.no_grad():
        for i in range(len(webpage_titles)):
            tokenized_text = tokenizer(webpage_titles[i], return_tensors="pt").to(device)
            output_base = model(**tokenized_text)
            logits_base = output_base.logits
            probs_base = F.softmax(logits_base, dim=-1)
            positive_probs_base = probs_base[0][1].item()
            model_outputs.append(positive_probs_base)
            
    # Save using pickle
    with open('model_outputs_base.pkl', 'wb') as f:
        pickle.dump(model_outputs, f)
    """
    with open("model_outputs.pkl", 'rb') as f:
        model_output = pickle.load(f)

    with open("synthetic_data.pkl", 'rb') as f:
        synthetic_data = pickle.load(f)
    
    with open("model_outputs_base.pkl", 'rb') as f:
        model_output_base = pickle.load(f)
        
        # Ensure data is properly formatted (flatten if needed)
    model_output = np.array(model_output).flatten()
    synthetic_data = np.array(synthetic_data).flatten()
    model_base_output = np.array(model_output_base).flatten()
    
    
    # Ensure they have the same length
    min_length = min(len(model_output), len(synthetic_data), len(model_base_output))
    model_output = model_output[:min_length]
    synthetic_data = synthetic_data[:min_length]
    model_base_output = model_base_output[:min_length]
    diff = model_base_output - model_output
    
      
    plt.hist(x=diff, bins=20, color="skyblue", edgecolor="black")
    plt.xlabel('Differences')
    plt.ylabel('Frequencies')
    plt.title('Frequencies v.s Differences')
    plt.show()
    plt.savefig("histogram_new.png")
    
    """
    mae = np.mean(np.abs(diff))
    print(mae)
    plt.scatter(model_output, model_base_output)
    plt.xlabel("Synthetic Model Output")
    plt.ylabel("Base Model Output")
    plt.title("Base v.s Synthetic")
    plt.show()
    plt.savefig("base_synthetic.png")
    
    
    # Calculate error
    Y = model_output - synthetic_data
    
    # Calculate MAE correctly
    mae = np.mean(np.abs(Y))
    print(f"Mean Absolute Error: {mae}")
    
    plt.hist(x=Y, bins=20, color="skyblue", edgecolor="black")
    plt.xlabel('Differences')
    plt.ylabel('Frequencies')
    plt.title('Frequencies v.s Differences')
    plt.show()
    plt.savefig("histogram.png")
    """
    
    
    