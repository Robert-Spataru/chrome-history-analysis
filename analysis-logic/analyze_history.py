import pandas as pd
import numpy as np
import torch
import nltk as tk  
from chrome_history import BrowserHistorySelector
from helper import format_file_path
import matplotlib.pyplot as plt
import sys
import os
import pickle

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sentiment_model_finetuning.model import *
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class BrowserHistoryAnalyzer:
    def __init__(self, dataframe_path, time_period):
        """
        Dataframe should have columns: title, visit_count, last_visit_time.
        analyzer_method: 1 for sentiment_over_time, 2 for word_semantics_over_time.
        """
        self.dataframe_path = dataframe_path
        self.time_period = time_period
        self.analyzer_method_menu = {
            1: "sentiment_over_time"
        }
    
    def run_history_selector(self, path_name_1, path_name_2):
        analyzer = BrowserHistorySelector()
        # Always use the single available query.
        result_df = analyzer.run_query(self.time_period)
        print(result_df.head())
        result_df.to_csv(f'data/{path_name_1}.csv', index=False)
        selected_columns = analyzer.get_important_info(result_df, ['title', 'visit_count', 'last_visit_time'])
        selected_columns.to_csv(f'data/{path_name_2}.csv', index=False)
        print(selected_columns.head())
        
    def get_sentiment_over_time(self):
        device = torch.device("cpu")
        model_name = "siebert/sentiment-roberta-large-english"
        model_finetuned = SentimentRegressionModelWithLoRA(model_name).to(device)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        lora_finetuned_path = os.path.join(current_dir, "..", "model-parameters", "LoRA_Finetuned.pt")
        #lora_finetuned_path = "model-parameters/LoRA_Finetuned.pt"
        model_finetuned.load_state_dict(torch.load(lora_finetuned_path, weights_only=True))
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        df = pd.read_csv(self.dataframe_path)
        titles = df["title"]
        times = df["last_visit_time"]
        times = np.array(times)
        sentiment_scores = []
        length = min(len(titles), len(times))
        with torch.no_grad():
            for i in range(length):
                tokenized_text = tokenizer(str(titles[i]), return_tensors="pt").to(device)
                model_output = model_finetuned(**tokenized_text)
                sentiment_score = torch.tensor(model_output[0], requires_grad=False) #positive_sentiment_score
                sentiment_scores.append(sentiment_score.detach().numpy())
            sentiment_scores = np.array(sentiment_scores)
        
        with open("times.pkl", "wb") as f:
            pickle.dump(times, f)
            
        with open("sentiment_scores.pkl", "wb") as f:
            pickle.dump(sentiment_scores, f)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(times, sentiment_scores, marker='.', linestyle='', markersize=4)
        plt.plot(times, sentiment_scores)
        plt.title("Positive Sentiment Scores v.s Times")
        plt.xlabel("Time")
        plt.ylabel("Positive Sentiment Score")
        plt.tight_layout()
        plt.show()
        plt.savefig("sentiment.png")
        
    
    def perform_analysis(self, analyzer_method):
        method = self.analyzer_method_menu[analyzer_method]
        if method == "sentiment_over_time":
            self.get_sentiment_over_time()
        elif method == "word_semantics_over_time":
            self.get_word_semantics_over_time()
        else:
            pass
def main():
    
    # Example time period: last month
    test_time_period = {'years': 0, 'months': 1, 'weeks': 0, 'days': 0, 'hours': 0, 'minutes': 0, 'seconds': 0}
    
    # Use a fixed file path for the selected columns.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_path = os.path.join(current_dir, "..", "data", "selected_columns.csv")
    
    # For example, use analyzer_method 1 (sentiment_over_time)
    analyzer_method = 1
    
    history_analyzer = BrowserHistoryAnalyzer(test_path, test_time_period)
    history_analyzer.perform_analysis(1)
        
if __name__ == "__main__":
    main()

        