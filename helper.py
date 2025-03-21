def format_file_path(base_path, query_number, time_period,):
    return base_path.format(query_number, time_period["years"], time_period["months"], time_period["weeks"], time_period["days"], time_period["hours"])

