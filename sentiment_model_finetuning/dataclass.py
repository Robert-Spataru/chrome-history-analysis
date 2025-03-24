import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class WebpageTitleSentiment(Dataset):
    def __init__(self, webpage_titles, sentiment_scores, tokenizer):
        self.webpage_titles = webpage_titles  # text of the webpage titles that users visit
        self.sentiment_scores = sentiment_scores  # ranging from 0.0 - 1.0
        self.tokenizer = tokenizer  # tokenizer for preprocessing
        
    def __len__(self):
        return len(self.webpage_titles)
        
    def __getitem__(self, idx):
        title = self.webpage_titles[idx]
        
        encoding = self.tokenizer(title, return_tensors="pt", padding="max_length", 
                                     truncation=True, max_length=128)
        # Remove the batch dimension
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
            
                # For regression: return the continuous score as is
        score = torch.tensor(self.sentiment_scores[idx], dtype=torch.float)
        return encoding, score


#filepath = synthetic_training_data/synthetic_data.csv
def get_data_from_filepath(filepath):
    dataframe = pd.read_csv(filepath)
    webpage_titles = dataframe["Title"].to_list()
    sentiment_scores = dataframe["Sentiment Score"].to_list()
    return webpage_titles, sentiment_scores


def get_training_loader(training_data, params):
    train_loader = DataLoader(training_data, **params)
    return train_loader

def get_testing_loader(testing_data, params):
    test_loader = DataLoader(testing_data, **params)
    return test_loader
