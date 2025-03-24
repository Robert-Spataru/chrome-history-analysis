import sqlite3
import os
import pandas as pd
from datetime import datetime, timedelta, timezone
import shutil

class BrowserHistorySelector:
    def __init__(self, chrome_path=None):
        """Initialize the Browser History Analyzer with an optional custom path"""
        self.chrome_history_path = chrome_path or os.path.expanduser(
            "~/Library/Application Support/Google/Chrome/Default/History"
        )
        self.temp_path = os.path.expanduser(
            "~/Library/Application Support/Google/Chrome/Default/History_copy"
        )
        self.tz = self.get_timezone_offset()
        # Hard-coded single query
        self.query = (
            "SELECT * FROM urls WHERE {} < last_visit_time AND last_visit_time < {} "
            "ORDER BY last_visit_time DESC;"
        )
        
    def get_timezone_offset(self):
        """Get timezone offset in hours"""
        current_datetime = datetime.now().replace(tzinfo=None)
        utc_time = datetime.now(timezone.utc).replace(tzinfo=None)
        datetime_offset = current_datetime - utc_time
        return round(datetime_offset.total_seconds() / 3600)
    
    def dt2str(self, dt):
        """Convert datetime to string format"""
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    
    def date_from_webkit(self, webkit_timestamp):
        """Convert webkit (UTC) to local datetime"""
        epoch_start = datetime(1601, 1, 1)
        delta = timedelta(hours=self.tz, microseconds=int(webkit_timestamp))
        return epoch_start + delta
    
    def date_to_webkit(self, dt):
        """Convert local datetime to webkit (UTC)"""
        epoch_start = datetime(1601, 1, 1)
        delta = dt - epoch_start - timedelta(hours=self.tz)
        delta_micro_sec = (delta.days * 86400 + delta.seconds) * 1000000
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
        shutil.copy2(self.chrome_history_path, self.temp_path)
        return self.temp_path
    
    def cleanup(self):
        """Remove temporary database file"""
        if os.path.exists(self.temp_path):
            os.remove(self.temp_path)
    
    def run_query(self, time_period=None, as_dataframe=True):
        """
        Run the hard-coded query with an optional time period.
        
        Parameters:
            time_period: Dictionary with keys 'years', 'months', 'weeks', 'days', 'hours', 'minutes', 'seconds'
                         Default: last 24 hours.
            as_dataframe: Return as pandas DataFrame (True) or list of tuples (False)
        
        Returns:
            DataFrame or list of results.
        """
        # Default to last 24 hours if no time period provided
        if time_period is None:
            time_period = {'years': 0, 'months': 0, 'weeks': 0, 'days': 1, 'hours': 0, 'minutes': 0, 'seconds': 0}
        
        start_date, end_date = self.get_time_range(**time_period)
        time_from = self.date_to_webkit(start_date)
        time_to = self.date_to_webkit(end_date)
        
        try:
            self.prepare_database()
            conn = sqlite3.connect(self.temp_path)
            formatted_query = self.query.format(time_from, time_to)
            
            if as_dataframe:
                df = pd.read_sql_query(formatted_query, conn)
                if 'last_visit_time' in df.columns:
                    df['last_visit_time'] = df['last_visit_time'].apply(self.date_from_webkit)
                result = df
            else:
                cursor = conn.cursor()
                cursor.execute(formatted_query)
                result = cursor.fetchall()
                cursor.close()
            
            conn.close()
            return result
        
        except Exception as e:
            raise Exception(f"Error executing query: {str(e)}")
        finally:
            self.cleanup()
    
    def get_important_info(self, dataframe, columns_list):
        """Extract selected columns (e.g., title, last_visit_time)"""
        return dataframe[columns_list]

def main():
    analyzer = BrowserHistorySelector()
    # Default to last 24 hours
    time_period = {'years': 0, 'months': 0, 'weeks': 0, 'days': 1, 'hours': 0, 'minutes': 0, 'seconds': 0}
    result_df = analyzer.run_query(time_period)
    print(result_df.head())
    result_df.to_csv('data/month_history.csv', index=False)
    selected_columns = analyzer.get_important_info(result_df, ['title', 'last_visit_time'])
    selected_columns.to_csv('data/selected_columns.csv', index=False)
    print(selected_columns.head())
    
if __name__ == "__main__":
    main()
