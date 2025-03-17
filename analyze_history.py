queries_dict = {
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
                                LIMIT 10;"""
}

