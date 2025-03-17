'''
Behavioral Insights: Correlate sentiment with time of day—do you visit happier sites in the morning?
Site Preferences: Rank domains by visit count and sentiment—do you prefer positive or negative content?
Habit Tracking: Identify peak browsing times or over-visited sites for personal reflection.
Recommendation Tool: Suggest reducing time on negative-sentiment sites based on your analysis.
'''
import sqlite3
import os
from datetime import date, datetime, timedelta, timezone
import shutil


#get time range given the unit (minutes, hours, days, months, years) and then the corrseponding quantities as a dictionary
def get_time_range(range_dictionary: dict) -> dict:
    years = (float)(range_dictionary["years"])
    months = (float)(range_dictionary["months"])
    weeks = (float)(range_dictionary["weeks"] + (months * 4) + (years * 52))
    days = (float)(range_dictionary["days"])
    hours = (float)(range_dictionary["hours"])
    seconds = (float)(range_dictionary["seconds"])
    current_date = datetime.now()
    time_delta = timedelta(weeks=weeks, days=days, hours=hours, seconds=seconds)
    start_date = current_date - time_delta
    return start_date, current_date

def get_timezone_offset():
    current_datetime = datetime.now().replace(tzinfo=None)
    utc_time = datetime.now(timezone.utc).replace(tzinfo=None)
    datetime_offset = current_datetime - utc_time
    datetime_offset_in_seconds = datetime_offset.total_seconds()
    datetime_offset_in_round_hours = round(datetime_offset_in_seconds / 3600)
    return datetime_offset_in_round_hours

def dt2str(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def date_from_webkit(webkit_timestamp, tz):
    """Convert webkit(utc) to local datetime"""
    epoch_start = datetime(1601, 1, 1)
    delta = timedelta(hours=tz, microseconds=int(webkit_timestamp))
    return epoch_start + delta

def date_to_webkit(dt, tz):
    """Convert local datetime to webkit(utc)"""
    epoch_start = datetime(1601, 1, 1)
    delta = dt - epoch_start - timedelta(hours=tz)
    delta_micro_sec = (delta.days * 60 * 60 * 24 + delta.seconds) * 1000 * 1000
    return delta_micro_sec

def time_range_set(dt_from, dt_to, tz):
    time_to = date_to_webkit(dt_to, tz)
    time_from = date_to_webkit(dt_from, tz)
    return time_from, time_to

def execute_query(chrome_history_path, sql_query, argument_dict):
    tz = get_timezone_offset()
    start_date = argument_dict["start_date"]
    current_date = argument_dict["current_date"]
    keyword = argument_dict["keyword"]
    time_from, time_to = time_range_set(start_date, current_date, tz)
    sql_query = sql_query.format(time_from, time_to)
    con = sqlite3.connect(chrome_history_path)
    cur = con.cursor()
    cur.execute(sql_query)
    results = cur.fetchall()
    cur.close()
    return results

def setup_query(range_dictionary, sql_query):
    argument_dict = {}
    
    src_path = os.path.expanduser("~/Library/Application Support/Google/Chrome/Default/History")
    if not os.path.exists(src_path):
        print("Path doesn't exist")
    temp_path = os.path.expanduser("~/Library/Application Support/Google/Chrome/Default/History_copy")

    shutil.copy2(src_path, temp_path)  # Copy file before querying
    start_date, current_date = get_time_range(range_dictionary)
    keyword = None #top titles by a given keyword
    argument_dict["start_date"] = start_date
    argument_dict["current_date"] = current_date
    argument_dict["keyword"] = keyword
    
    results = execute_query(temp_path, sql_query, argument_dict)
    return results
    



def main():
    years = 0
    months = 0
    weeks = 0
    days = 0
    hours = 2
    minutes = 0
    seconds = 0
    range_dictionary = {"years":years, "months":months, "weeks":weeks, "days":days, "hours":hours, "minutes":minutes, "seconds":seconds}
    sql_query = """SELECT * FROM urls
                    WHERE {} < last_visit_time
                    AND last_visit_time < {}
                    ORDER BY last_visit_time DESC;"""
    
    
    
    
    
    

if __name__ == "__main__":
    main()