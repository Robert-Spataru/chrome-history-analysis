'''
Behavioral Insights: Correlate sentiment with time of day—do you visit happier sites in the morning?
Site Preferences: Rank domains by visit count and sentiment—do you prefer positive or negative content?
Habit Tracking: Identify peak browsing times or over-visited sites for personal reflection.
Recommendation Tool: Suggest reducing time on negative-sentiment sites based on your analysis.
'''
import sqlite3
import os
import pandas as pd
from datetime import date, datetime, timedelta, timezone
import shutil
import json

class BrowserHistorySelector:
    def __init__(self, chrome_path=None):
        """Initialize the Browser History Analyzer with optional custom path"""
        self.chrome_history_path = chrome_path or os.path.expanduser("~/Library/Application Support/Google/Chrome/Default/History")
        self.temp_path = os.path.expanduser("~/Library/Application Support/Google/Chrome/Default/History_copy")
        self.tz = self.get_timezone_offset()
        
        # Dictionary of available queries
        self.queries = {
            "most_recent_query": """SELECT * FROM urls WHERE {} < last_visit_time 
                                AND last_visit_time < {} ORDER BY last_visit_time DESC;""",
            
            "top_webpage_visits": """SELECT url, title, visit_count, last_visit_time FROM urls 
                                 WHERE {} < last_visit_time AND last_visit_time < {} 
                                 ORDER BY visit_count DESC LIMIT 10;""",
            
            "most_frequent_domains": """SELECT SUBSTRING(url, 'https?://([^/]+)') AS domain, 
                                    SUM(visit_count) AS total_visits 
                                    FROM urls 
                                    WHERE {} < last_visit_time AND last_visit_time < {} 
                                    GROUP BY domain 
                                    ORDER BY total_visits DESC 
                                    LIMIT 5;""",
            
            "longest_revisit_gaps": """SELECT url, title, last_visit_time, 
                                   LAG(last_visit_time) OVER (PARTITION BY url ORDER BY last_visit_time) AS prev_visit, 
                                   (last_visit_time - LAG(last_visit_time) OVER (PARTITION BY url ORDER BY last_visit_time)) AS time_gap 
                                   FROM urls 
                                   WHERE {} < last_visit_time AND last_visit_time < {} 
                                   ORDER BY time_gap DESC 
                                   LIMIT 10;""",
            
            "single_visit_pages": """SELECT url, title, last_visit_time 
                                 FROM urls 
                                 WHERE {} < last_visit_time AND last_visit_time < {} 
                                 AND visit_count = 1 
                                 ORDER BY last_visit_time DESC;""",
            
            "most_active_day": """SELECT DATE(last_visit_time) AS visit_day, 
                              COUNT(*) AS visit_count 
                              FROM urls 
                              WHERE {} < last_visit_time AND last_visit_time < {} 
                              GROUP BY visit_day 
                              ORDER BY visit_count DESC 
                              LIMIT 1;""",
            
            "recent_title_changes": """SELECT url, title, last_visit_time, 
                                   LAG(title) OVER (PARTITION BY url ORDER BY last_visit_time) AS prev_title 
                                   FROM urls 
                                   WHERE {} < last_visit_time AND last_visit_time < {} 
                                   AND title != LAG(title) OVER (PARTITION BY url ORDER BY last_visit_time) 
                                   ORDER BY last_visit_time DESC 
                                   LIMIT 10;""",
            
            "shortest_visit_intervals": """SELECT url, title, last_visit_time, 
                                       (last_visit_time - LAG(last_visit_time) OVER (PARTITION BY url ORDER BY last_visit_time)) AS interval 
                                       FROM urls 
                                       WHERE {} < last_visit_time AND last_visit_time < {} 
                                       AND visit_count > 1 
                                       ORDER BY interval ASC 
                                       LIMIT 10;""",
            
            "top_titles_by_keyword": """SELECT url, title, visit_count, last_visit_time 
                                    FROM urls 
                                    WHERE {} < last_visit_time AND last_visit_time < {} 
                                    AND title LIKE '%{}%' 
                                    ORDER BY visit_count DESC 
                                    LIMIT 10;""",
            
            # Pattern-focused queries for ML sentiment analysis
            "temporal_search_patterns": """SELECT
                                       DATE(last_visit_time) AS visit_date,
                                       EXTRACT(HOUR FROM last_visit_time) AS hour_of_day,
                                       COUNT(*) AS search_count,
                                       AVG(LENGTH(title)) AS avg_title_length,
                                       COUNT(DISTINCT SUBSTRING(url, 'https?://([^/]+)')) AS unique_domains_visited
                                    FROM urls
                                    WHERE {} < last_visit_time AND last_visit_time < {}
                                    GROUP BY visit_date, hour_of_day
                                    ORDER BY visit_date, hour_of_day;""",
            
            "search_velocity": """SELECT
                               DATE(last_visit_time) AS visit_date,
                               COUNT(*) AS total_searches,
                               COUNT(*) / 24.0 AS searches_per_hour,
                               MAX(last_visit_time) - MIN(last_visit_time) AS active_period,
                               COUNT(*) / (EXTRACT(EPOCH FROM (MAX(last_visit_time) - MIN(last_visit_time)))/3600) AS search_velocity
                            FROM urls
                            WHERE {} < last_visit_time AND last_visit_time < {}
                            GROUP BY visit_date
                            ORDER BY visit_date;""",
            
            "query_evolution": """SELECT
                               url,
                               title,
                               last_visit_time,
                               LENGTH(title) AS title_length,
                               LAG(LENGTH(title)) OVER (PARTITION BY SUBSTRING(url, 'https?://([^/]+)') ORDER BY last_visit_time) AS prev_title_length,
                               EXTRACT(EPOCH FROM (last_visit_time - LAG(last_visit_time) OVER (PARTITION BY SUBSTRING(url, 'https?://([^/]+)') ORDER BY last_visit_time)))/60 AS minutes_since_last_visit
                            FROM urls
                            WHERE {} < last_visit_time AND last_visit_time < {}
                            AND LAG(last_visit_time) OVER (PARTITION BY SUBSTRING(url, 'https?://([^/]+)') ORDER BY last_visit_time) IS NOT NULL
                            ORDER BY last_visit_time;""",
            
            "search_sessionization": """WITH search_sessions AS (
                                    SELECT
                                        url,
                                        title,
                                        last_visit_time,
                                        SUBSTRING(url, 'https?://([^/]+)') AS domain,
                                        CASE
                                            WHEN EXTRACT(EPOCH FROM (last_visit_time - 
                                                LAG(last_visit_time) OVER (ORDER BY last_visit_time))) > 1800 
                                            THEN 1
                                            ELSE 0
                                        END AS new_session
                                    FROM urls
                                    WHERE {} < last_visit_time AND last_visit_time < {}
                                    ORDER BY last_visit_time
                                    )
                                    SELECT
                                        SUM(new_session) OVER (ORDER BY last_visit_time) AS session_id,
                                        url,
                                        title,
                                        last_visit_time,
                                        domain
                                    FROM search_sessions
                                    ORDER BY last_visit_time;""",
            
            "domain_switching_behavior": """WITH domain_visits AS (
                                        SELECT
                                            SUBSTRING(url, 'https?://([^/]+)') AS domain,
                                            last_visit_time,
                                            LAG(SUBSTRING(url, 'https?://([^/]+)')) OVER (ORDER BY last_visit_time) AS prev_domain
                                        FROM urls
                                        WHERE {} < last_visit_time AND last_visit_time < {}
                                        ORDER BY last_visit_time
                                        )
                                        SELECT
                                            DATE(last_visit_time) AS visit_date,
                                            COUNT(*) AS total_visits,
                                            COUNT(CASE WHEN domain != prev_domain AND prev_domain IS NOT NULL THEN 1 END) AS domain_switches,
                                            COUNT(CASE WHEN domain != prev_domain AND prev_domain IS NOT NULL THEN 1 END)::FLOAT / NULLIF(COUNT(*), 0) AS switching_rate
                                        FROM domain_visits
                                        GROUP BY visit_date
                                        ORDER BY visit_date;""",
            
            "query_refinement_patterns": """WITH sequential_queries AS (
                                        SELECT
                                            title,
                                            last_visit_time,
                                            LAG(title) OVER (ORDER BY last_visit_time) AS prev_title,
                                            LAG(last_visit_time) OVER (ORDER BY last_visit_time) AS prev_time
                                        FROM urls
                                        WHERE {} < last_visit_time AND last_visit_time < {}
                                        ORDER BY last_visit_time
                                        )
                                        SELECT
                                            DATE(last_visit_time) AS visit_date,
                                            COUNT(*) AS total_queries,
                                            COUNT(CASE WHEN 
                                                EXTRACT(EPOCH FROM (last_visit_time - prev_time)) < 60 AND
                                                (title LIKE '%' || prev_title || '%' OR prev_title LIKE '%' || title || '%')
                                                THEN 1 END) AS refined_queries,
                                            COUNT(CASE WHEN 
                                                EXTRACT(EPOCH FROM (last_visit_time - prev_time)) < 60 AND
                                                NOT (title LIKE '%' || prev_title || '%' OR prev_title LIKE '%' || title || '%')
                                                THEN 1 END) AS topic_shifts
                                        FROM sequential_queries
                                        WHERE prev_title IS NOT NULL
                                        GROUP BY visit_date
                                        ORDER BY visit_date;""",
            
            "daily_search_rhythm": """SELECT
                                   DATE(last_visit_time) AS visit_date,
                                   EXTRACT(HOUR FROM last_visit_time) AS hour_of_day,
                                   COUNT(*) AS search_count,
                                   AVG(LENGTH(title)) AS avg_query_length,
                                   COUNT(DISTINCT SUBSTRING(url, 'https?://([^/]+)')) AS unique_domains,
                                   MAX(last_visit_time) - MIN(last_visit_time) AS time_span
                                FROM urls
                                WHERE {} < last_visit_time AND last_visit_time < {}
                                GROUP BY visit_date, hour_of_day
                                ORDER BY visit_date, hour_of_day;""",
            
            "topic_distribution": """WITH url_topics AS (
                                SELECT
                                    url,
                                    title,
                                    last_visit_time,
                                    CASE
                                        WHEN url LIKE '%google%' OR url LIKE '%bing%' OR url LIKE '%search%' THEN 'search_engine'
                                        WHEN url LIKE '%facebook%' OR url LIKE '%twitter%' OR url LIKE '%instagram%' OR url LIKE '%linkedin%' THEN 'social_media'
                                        WHEN url LIKE '%amazon%' OR url LIKE '%ebay%' OR url LIKE '%shop%' OR url LIKE '%buy%' THEN 'shopping'
                                        WHEN url LIKE '%youtube%' OR url LIKE '%netflix%' OR url LIKE '%hulu%' OR url LIKE '%stream%' THEN 'entertainment'
                                        WHEN url LIKE '%news%' OR url LIKE '%cnn%' OR url LIKE '%bbc%' OR url LIKE '%nyt%' THEN 'news'
                                        WHEN url LIKE '%github%' OR url LIKE '%stack%' OR url LIKE '%code%' OR url LIKE '%dev%' THEN 'programming'
                                        WHEN url LIKE '%scholar%' OR url LIKE '%edu%' OR url LIKE '%learn%' OR url LIKE '%course%' THEN 'educational'
                                        ELSE 'other'
                                    END AS topic_category
                                FROM urls
                                WHERE {} < last_visit_time AND last_visit_time < {}
                                )
                                SELECT
                                    DATE(last_visit_time) AS visit_date,
                                    topic_category,
                                    COUNT(*) AS category_count,
                                    ROUND(COUNT(*)::FLOAT / SUM(COUNT(*)) OVER (PARTITION BY DATE(last_visit_time)) * 100, 2) AS percentage
                                FROM url_topics
                                GROUP BY visit_date, topic_category
                                ORDER BY visit_date, category_count DESC;"""
        }
        
    def get_timezone_offset(self):
        """Get timezone offset in hours"""
        current_datetime = datetime.now().replace(tzinfo=None)
        utc_time = datetime.now(timezone.utc).replace(tzinfo=None)
        datetime_offset = current_datetime - utc_time
        datetime_offset_in_seconds = datetime_offset.total_seconds()
        datetime_offset_in_round_hours = round(datetime_offset_in_seconds / 3600)
        return datetime_offset_in_round_hours
    
    def dt2str(self, dt):
        """Convert datetime to string format"""
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    
    def date_from_webkit(self, webkit_timestamp):
        """Convert webkit(utc) to local datetime"""
        epoch_start = datetime(1601, 1, 1)
        delta = timedelta(hours=self.tz, microseconds=int(webkit_timestamp))
        return epoch_start + delta
    
    def date_to_webkit(self, dt):
        """Convert local datetime to webkit(utc)"""
        epoch_start = datetime(1601, 1, 1)
        delta = dt - epoch_start - timedelta(hours=self.tz)
        delta_micro_sec = (delta.days * 60 * 60 * 24 + delta.seconds) * 1000 * 1000
        return delta_micro_sec
    
    def get_time_range(self, years=0, months=0, weeks=0, days=0, hours=0, minutes=0, seconds=0):
        """Calculate time range based on specified time units"""
        total_weeks = weeks + (months * 4) + (years * 52)
        current_date = datetime.now()
        time_delta = timedelta(weeks=total_weeks, days=days, hours=hours, minutes=minutes, seconds=seconds)
        start_date = current_date - time_delta
        return start_date, current_date
    
    def prepare_database(self):
        """Make a copy of the Chrome history database to avoid conflicts"""
        if not os.path.exists(self.chrome_history_path):
            raise FileNotFoundError(f"Chrome history file not found at {self.chrome_history_path}")
        
        # Create a copy of the database to avoid conflicts with Chrome
        shutil.copy2(self.chrome_history_path, self.temp_path)
        return self.temp_path
    
    def cleanup(self):
        """Remove temporary database file"""
        if os.path.exists(self.temp_path):
            os.remove(self.temp_path)
    
    def run_query(self, query_name, time_period=None, keyword=None, as_dataframe=True):
        """
        Run a predefined query by name with optional time period
        
        Parameters:
        - query_name: Name of the query from self.queries
        - time_period: Dict with keys 'years', 'months', 'weeks', 'days', 'hours', 'minutes', 'seconds'
                    Default: last 24 hours
        - keyword: Optional keyword for searches that require it
        - as_dataframe: Return as pandas DataFrame (True) or list of tuples (False)
        
        Returns:
        - DataFrame or list of results
        """
        # Default to last 24 hours if no time period provided
        if time_period is None:
            time_period = {'years': 0, 'months': 0, 'weeks': 0, 'days': 1, 'hours': 0, 'minutes': 0, 'seconds': 0}
        
        # Validate that query exists
        if query_name not in self.queries:
            raise ValueError(f"Query '{query_name}' not found. Available queries: {', '.join(self.queries.keys())}")
        
        # Get the SQL query template
        sql_query = self.queries[query_name]
        
        # Calculate date range
        start_date, end_date = self.get_time_range(
            years=time_period.get('years', 0),
            months=time_period.get('months', 0),
            weeks=time_period.get('weeks', 0),
            days=time_period.get('days', 0),
            hours=time_period.get('hours', 0),
            minutes=time_period.get('minutes', 0),
            seconds=time_period.get('seconds', 0)
        )
        
        # Convert dates to webkit timestamp format
        time_from = self.date_to_webkit(start_date)
        time_to = self.date_to_webkit(end_date)
        
        try:
            # Prepare database (copy to avoid conflict with Chrome)
            self.prepare_database()
            
            # Connect to the database
            conn = sqlite3.connect(self.temp_path)
            
            # Format the query with the time range and optional keyword
            if 'keyword' in query_name and keyword:
                formatted_query = sql_query.format(time_from, time_to, keyword)
            else:
                formatted_query = sql_query.format(time_from, time_to)
            
            # Execute the query and fetch results
            if as_dataframe:
                # Return as pandas DataFrame
                df = pd.read_sql_query(formatted_query, conn)
                
                # Convert webkit timestamps to readable dates if present
                if 'last_visit_time' in df.columns:
                    df['last_visit_time'] = df['last_visit_time'].apply(lambda x: self.date_from_webkit(x))
                
                result = df
            else:
                # Return as list of tuples
                cursor = conn.cursor()
                cursor.execute(formatted_query)
                result = cursor.fetchall()
                cursor.close()
            
            conn.close()
            return result
        
        except Exception as e:
            raise Exception(f"Error executing query: {str(e)}")
        
        finally:
            # Clean up temporary file
            self.cleanup()
            
    def get_important_info(self, dataframe, columns_list):
        """extract tile, visit_count, and last_visit_time"""
        selected_columns = dataframe[columns_list]
        return selected_columns
    
            
def main():
    analyzer = BrowserHistorySelector()
    time_period = {'years': 0, 'months': 1, 'weeks': 0, 'days': 0, 'hours': 0, 'minutes': 0, 'seconds': 0}
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
    query_number = 1
    result_df = analyzer.run_query(query_menu[query_number], time_period)
    print(result_df.head())
    result_df.to_csv('data/month_history.csv', index=False)
    columns_list = ['title', 'visit_count', 'last_visit_time']
    selected_columns = analyzer.get_important_info(result_df, columns_list)
    selected_columns.to_csv('data/selected_columns.csv', index=False)
    print(selected_columns.head())
    

if __name__ == "__main__":
    #main()
    frame = pd.read_csv("data/month_history.csv")
    print(frame.head())