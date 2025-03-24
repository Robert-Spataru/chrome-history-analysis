from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch
import torch.nn.functional as F
import torch.nn as nn
from dataclass import *
from model import *
from tqdm.auto import tqdm
import os


negative_text = "Man kills a women with a knife on the highway"
medium_text = "The hidden pyramids of giza and their significance"
positive_text = "Student creates social media account that saves hundreds of lives"
model_name = "siebert/sentiment-roberta-large-english"
text_list = [negative_text, medium_text, positive_text]

tokenizer = AutoTokenizer.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_base = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
model_base.eval()

model_finetuned = SentimentRegressionModelWithLoRA(model_name).to(device)
current_dir = os.path.dirname(os.path.abspath(__file__))
lora_finetuned_path = os.path.join(current_dir, "..", "model-parameters", "LoRA_Finetuned.pt")
model_finetuned.load_state_dict(torch.load(lora_finetuned_path, weights_only=True))
model_finetuned.eval()


tokenized_text_list = []

for text in text_list:
    tokenized_text = tokenizer(text, return_tensors="pt").to(device)
    print(tokenized_text)
    print(type(tokenized_text))
    #tokenized_text_temp = {key: value.to(device) for key, value in tokenized_text}
    #print(tokenized_text_temp)
    #print(type(tokenized_text_temp))
    tokenized_text_list.append(tokenized_text)
    output_base = model_base(**tokenized_text)
    logits_base = output_base.logits
    probs_base = F.softmax(logits_base, dim=-1)
    negative_probs_base = probs_base[0][0].item()
    positive_probs_base = probs_base[0][1].item()
    print(f"The following text: \"{text}\" has a {positive_probs_base} probability of having positive sentiment and a {negative_probs_base} probability of having negative sentiment")
    
    output_finetuned = model_finetuned(**tokenized_text)
    positive_probs_finetuned = output_finetuned[0]
    negative_probs_finetuned = 1.0 - positive_probs_finetuned
    print(f"The following text: \"{text}\" has a {positive_probs_finetuned} probability of having positive sentiment and a {negative_probs_finetuned} probability of having negative sentiment")
    
    
#sentiment_analysis = pipeline("sentiment-analysis",model="siebert/sentiment-roberta-large-english", device=device)
#print(sentiment_analysis("I love this!"))

"""
#proper testing
filepath = "synthetic_training_data/synthetic_data.csv"
webpage_titles, sentiment_scores = get_data_from_filepath(filepath)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load datasets
dataset = WebpageTitleSentiment(webpage_titles, sentiment_scores, tokenizer)

# Create test data loader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
"""