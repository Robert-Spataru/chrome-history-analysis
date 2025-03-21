import pandas as pd
import ntlk as tk
from chrome_history import *
from helper import *
import matplotlib as plt

class BrowserHistoryAnalyzer():
    def __init__(self, dataframe_path, analyzer_method, time_period, query_menu, query_number):
        """data should have the format of title, visit_count, last_visit_time"""
        self.dataframe_path = dataframe_path
        self.method = analyzer_method
        self.time_period = time_period
        self.query_menu = query_menu
        self.query_number = query_number
        self.analyzer_method_menu = {
        1: "sentiment_over_time",
        2: "word_semantics_over_time",
    }
    
    def run_history_selector(self, path_name_1, path_name_2):
        analyzer = BrowserHistorySelector()
        result_df = analyzer.run_query(self.query_menu[self.query_number], self.time_period)
        print(result_df.head())
        result_df.to_csv('data/{path_name_1}.csv', index=False)
        columns_list = ['title', 'visit_count', 'last_visit_time']
        selected_columns = analyzer.get_important_info(result_df, columns_list)
        selected_columns.to_csv('data/{path_name_2}.csv', index=False)
        print(selected_columns.head())
        
    
    def get_sentiment_over_time(self):
        df = pd.read_csv(self.dataframe_path)
        titles = df["titles"]
        visit_counts = df["visit_counts"]
        #calculate visit_count_weight
        visit_count_weight = self.calculate_visit_count_weight()
        
        vectorized_titles = []
        
        
        
        
    def get_word_semantics_over_time(self):
        pass
    
    def perform_analysis(self):
        method = self.analyzer_method_menu[self.method]
        match method:
            case "sentiment_over_time":
                self.get_sentiment_over_time()
            case "word_semantics_over_time":
                self.get_word_semantics_over_time()
            
    
    

def main():
    query_menu = {
    # Basic browsing history queries
    1: "most_recent_query",
    2: "top_webpage_visits",
    3: "most_frequent_domains",
    4: "longest_revisit_gaps",
    5: "single_visit_pages",
    6: "most_active_day",
    7: "recent_title_changes",
    8: "shortest_visit_intervals",
    9: "top_titles_by_keyword",
    
    # Pattern-focused queries for ML sentiment analysis
    10: "temporal_search_patterns",
    11: "search_velocity",
    12: "query_evolution",
    13: "search_sessionization",
    14: "domain_switching_behavior",
    15: "query_refinement_patterns",
    16: "daily_search_rhythm",
    17: "topic_distribution"
    }
    
    analyzer_method_menu = {
        1: "sentiment_over_time",
        2: "word_semantics_over_time",
    }
    
    query_number = 1

   
    time_period_1 = {'years': 1, 'months': 0, 'weeks': 0, 'days': 0, 'hours': 0, 'minutes': 0, 'seconds': 0}
    time_period_2 = {'years': 0, 'months': 1, 'weeks': 0, 'days': 0, 'hours': 0, 'minutes': 0, 'seconds': 0}
    time_period_3 = {'years': 0, 'months': 0, 'weeks': 1, 'days': 0, 'hours': 0, 'minutes': 0, 'seconds': 0}
    
    base_path = "data/{}{}{}{}{}{}"
    
    path_1 = format_file_path(base_path, query_number, time_period_1)
    path_2 = format_file_path(base_path, query_number, time_period_2)
    path_3 = format_file_path(base_path, query_number, time_period_3)
    
    analyzer_method_1 = 1
    analyzer_method_2 = 2
    
    test_path = "data/selected_columns.csv"
    test_time_period = time_period_2 = {'years': 0, 'months': 1, 'weeks': 0, 'days': 0, 'hours': 0, 'minutes': 0, 'seconds': 0}
    
    
    
    history_analyzer = BrowserHistoryAnalyzer(test_path, analyzer_method_1, test_time_period, query_menu, query_number)
    analysis = history_analyzer.perform_analysis()
    
        
if __name__ == "__main__":
    main()     
        