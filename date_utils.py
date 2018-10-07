import pandas as pd
import datetime as dt


# get the day from now
def get_date(days_ago, date_object=False):
    days = dt.datetime.now() - dt.timedelta(days=days_ago)
    print(days)

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
