import pandas as pd
import numpy as np
import dask
import dask.dataframe as dd
import io
import json
from python_utils import sqlite_utils as dbx
from python_utils import file_utils as fx
from python_utils import date_utils as dx


# check if it is a pandas or dask dataframe
def check_df(df):
    if isinstance(df, pd.DataFrame):
        return df
    else:
        return df.compute(get=dask.local.get_sync)


# read DB table into a pandas dataframe
def db_pd(db, table):
    conn = dbx.connect_db(db)
    query = "select * from " + table + ";"
    df = pd.read_sql_query(query, conn)
    return df


# create DB table from pandas dataframe
# mode: if_exists : {‘fail’, ‘replace’, ‘append’}, default ‘fail’
def df_db(db, table, df, mode, index):
    conn = dbx.connect_db(db)
    df.to_sql(table, conn, if_exists=mode, index=index)


# export db to json
'''
orient : string
    Indication of expected JSON string format.

        Series
            default is ‘index’
            allowed values are: {‘split’,’records’,’index’}

        DataFrame
            default is ‘columns’
            allowed values are: {‘split’,’records’,’index’,’columns’,’values’}

        The format of the JSON string
            ‘split’ : dict like {‘index’ -> [index], ‘columns’ -> [columns], ‘data’ -> [values]}
            ‘records’ : list like [{column -> value}, … , {column -> value}]
            ‘index’ : dict like {index -> {column -> value}}
            ‘columns’ : dict like {column -> {index -> value}}
            ‘values’ : just the values array
            ‘table’ : dict like {‘schema’: {schema}, ‘data’: {data}} describing the data, and the data component is like orient='records'.
'''


def db_json(db, table, path, orient, index):
    df = db_pd(db, table)
    df.to_json(path, orient=orient, index=index)


def select_sql_pd(db, table, fields, field, value):
    conn = dbx.connect_db(db)

    if type(fields) == list:
        fields = ", ".join(fields)

    value = "'%" + value + "%'"

    query = "SELECT %s FROM %s WHERE %s LIKE %s;" % (fields, table, field, value)

    try:
        df = pd.read_sql_query(query, conn)
        return df

    except Exception as e:
        print(e)
        pass



def get_field_value(db, table, field1, field2, value):

    try:
        df = select_sql_pd(db, table, [field1, field2], field2, value)

        if len(df) > 0:
            return df.ix[0][field1]
        else:
            pass

    except AttributeError as e:
        print(e)
        pass

    except Exception as e:
        print(e)
        pass


@fx.timer
def db_ddf(db, table, columns, partitions, chunksize, offset):
    conn = dbx.connect_db(db)
    df = pd.DataFrame()

    while True:
        query = "SELECT * FROM %s limit %s offset %s;" % (table, chunksize, offset)
        df = pd.read_sql_query(query, conn)
        ddt = dd.from_pandas(df[columns], npartitions=partitions)
        if offset == 0:
            final = ddt
        else:
            final = dd.concat([ddt, final], axis=0, interleave_partitions=True)

        offset += chunksize

        if df.shape[0] < chunksize:
            break

    print('table ' + table + ' loaded into dask dataframe')
    return final


def df_ddf(df, partitions):
    df = dd.from_pandas(df, npartitions=partitions)
    return df


def ddf_concat(df1, df2):
    df = dd.concat([df1, df2], axis=0, interleave_partitions=True)
    return df


# update sql field from dataframe column
@fx.timer
def df_update_sql_field(db, table, id, field, df, type):

    df = check_df(df)

    data = []

    for ix in df.to_records():
        data.append((
            str(ix[field]), str(ix[id])
        ))

    dbx.update_field(db, table, id, field, type, data)


def df_delete_sql_rows(db, table, id, df, type):

    df = check_df(df)

    data = []

    for ix in df.to_records():
        data.append(str(ix[id]))

    dbx.delete_rows(db, table, id, type, data)


def df_fn_sql(db, table, fn, fields, type, file, null=False):
    '''
    map a db field and create and update a field in db from that mapping
    field[0]: id of row in table
    field[0:]: fields to create/update
    '''

    df = fx.pickle_fn(lambda:  db_ddf(db, table, fields, 500, 50000, 0), 'dataframes', file)

    if null:
        # select only empty rows
        # df = df[df[fields[2]].isnull()]
        df = df[df[fields[2]] == '']

    df[fields[2]] = df[fields[1]].map(fn)

    df_update_sql_field(db, table, fields[0], fields[2], df, type)


# import json file to dataframe
def json_pd(fname, orient):
    with io.open(fname, encoding='utf-8') as f:
        data = json.load(f)

    df = pd.read_json(data, orient=orient)

    return df


# save dataframe to csv
def save_to_csv(df, fname):
        df.to_csv(fname + '.csv', sep=',', encoding='utf-8', index=False)


@fx.timer
def replace_none_df(df, field):
    # df[field] = df[field].map(lambda x: np.where(x.isnull(), 'None', x))
    df = check_df(df)
    df[field] = df[field].map(lambda x: np.where(x == 'nan', 'None', x))
    df[field] = df[field].map(lambda x: np.where(x == '', 'None', x))
    return df


@fx.timer
def clean_df(df, field):
    df[field] = df[field].astype(str)
    df = df[~df[field].isnull()]
    df = df[df[field] != 'nan']
    df = df[df[field] != '']
    return df


@fx.timer
def filter_df(df, type, field, value):
    df[field] = df[field].astype(type)
    df = df[df[field] == value]
    return df


def sort_df(db, table, field, head):
    df = db_ddf(db, table, [field], 500, 50000, 0)
    # df = df.groupby(field)[field].apply(lambda x: x.value_counts())
    df = df.groupby(field)[field].agg({field: 'count'}).compute()
    df = df.reset_index(name='count')
    df = df.sort_values(['count'], ascending=False)
    df = df.head(head)
    return df


def get_unique(df, field):
    unique = df[field].unique()
    result = 'Number of unique {}: {}'.format(field, len(unique))
    return result


def get_value_counts(df, field):
    df = df.groupby(field)[field].apply(lambda x: x.value_counts())
    return check_df(df)


@fx.timer
def groupby_count(df, fields):
    df = df.groupby(fields[0])[fields[1]].apply(lambda x: x.value_counts())
    return check_df(df)


@fx.timer
def groupby_size(df, fields):
    df = check_df(df)
    df = pd.crosstab(fields)
    # df = df.groupby(fields).size().unstack(fill_value=0)
    return df


@fx.timer
def count_field(df, field):
    df = df[field].value_counts()
    return check_df(df)


@fx.timer
def groupby_sum(df, fields):
    df = df.groupby(fields[0])[fields[1]].sum()
    return check_df(df)


def sum_multiple_df(df, fields):

    for field in fields[1:3]:
        df[field] = df[field].map(lambda x: pd.to_numeric(x, errors='coerce'))

    df[fields[3]] = df[fields[3]].map(lambda x: np.where(x == 'True', 1, 0))

    df = df.groupby(fields[0]).agg({fields[0]:'count', fields[1]:'sum', fields[2]:'sum', fields[3]:'sum'}).rename(columns={'domain': 'count'}).reset_index()

    return check_df(df)


# get a number of items from db per month
def get_tweets_by_month(start, end, num, db, table, fields, date_field):
        final = pd.DataFrame()
        for i in dx.date_range_month(start, end):
            df = dbx.select_sql_pd_limit(db, table, fields, date_field, i, num)
            final = pd.concat([df, final])
        return df
