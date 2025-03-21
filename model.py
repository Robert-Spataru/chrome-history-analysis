from transformers import pipeline
from torch.utils.data import Dataset, DataLoader

from tqdm.auto import tqdm

class WebpageTileSentiment(Dataset):
    def __init__(self, webpage_titles, sentiment_scores):
        self.webpage_titles = webpage_titles #text of the webpage title that user visits
        self.sentiment_scores = sentiment_scores #ranging from 0.0 - 1.0
        
    def __len__(self):
        return len(self.webpage_titles)
    
    def __getitem__(self, idx):
        return tuple(self.webpage_titles[idx], self.sentiment_scores[idx])
    
#train_loader = Dataloader(training_data, batch_size=64, shuffle=True)
#test_loader = DataLoader(testing_data, batch_size=64, shuffle=True)
sentiment_analysis = pipeline("sentiment-analysis",model="siebert/sentiment-roberta-large-english")

print(sentiment_analysis("I love this!"))