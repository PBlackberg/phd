import datetime
def convert_to_datetime(cftime_dates):
    result = []
    for date in cftime_dates:
        year, month, day = date.year, date.month, date.day
        if month == 2 and day > 29:
            continue
        result.append(datetime.datetime(year, month, day))



















