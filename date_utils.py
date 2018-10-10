import pandas as pd
import datetime as dt


# get the day from now
def get_date(days_ago, date_object=False):
    days = dt.datetime.now() - dt.timedelta(days=days_ago)

    if date_object:
        return days
    else:
        return '{0}-{1:0>2}-{2:0>2}'.format(days.year, days.month, days.day)


# get a list of days between two dates
def date_range_day(start, end, delta):
    return [get_date(i) for i in range(abs((end - start).days) + delta)]


# get a list of months between two dates
# date aliases here https://pandas.pydata.org/pandas-docs/stable/timeseries.html#timeseries-offset-aliases
def date_range_month(start, end, format, freq):
    return [i.strftime(format) for i in pd.date_range(start=start, end=end, freq=freq)]


# date formats at https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
def format_date_str(string, old_format, new_format):
    return dt.datetime.strptime('Sun Oct 07 23:59:33 +0000 2018', old_format).strftime(new_format)
