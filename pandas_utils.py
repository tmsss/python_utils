import pandas as pd
import numpy as np
import dask
import dask.dataframe as dd
import io
import json
import itertools

import sqlite_utils as dbx
import file_utils as fx
import date_utils as dx
import calc_utils as cx


# check if it is a pandas or dask dataframe
def check_df(df):
    if isinstance(df, pd.DataFrame):
        return df
    else:
        return df.compute()


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

    value = "'%" + str(value) + "%'"

    query = "SELECT %s FROM %s WHERE %s LIKE %s;" % (fields, table, field, str(value))

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
def db_ddf(db, table, columns, partitions, chunksize, offset=0):
    '''
    Load big sql table into dask dataframe in chunks to prevent memory exhaustion

    args
    ----
    db (str): database to connect
    table (str): database table
    columns (list): list of table columns to retrieve
    partitions (int): Number of dask partitions to use
    chunksize (int): Number of rows to return in each iteration of the sql query (affects memory allocated)
    offset (int): Offset rows in query (needed for sql query iteration, default=0)

    returns
    ----
    final (object): dask dataframe
    '''
    
    conn = dbx.connect_db(db)
    df = pd.DataFrame()

    while True:
        query = "SELECT * FROM {} limit {} offset {};".format(table, chunksize, offset)
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


@fx.timer
def db_ddf_limit(db, table, columns, partitions, limit):
    conn = dbx.connect_db(db)
    df = pd.DataFrame()

    query = "SELECT * FROM %s limit %s;" % (table, limit)
    df = pd.read_sql_query(query, conn)
    ddt = dd.from_pandas(df[columns], npartitions=partitions)

    print('table ' + table + ' loaded into dask dataframe')
    return ddt


@fx.timer
def db_ddf_limit_offset(db, table, columns, partitions, limit, offset):
    conn = dbx.connect_db(db)
    df = pd.DataFrame()

    query = "SELECT * FROM %s limit %s offset %s;" % (table, limit, offset)
    df = pd.read_sql_query(query, conn)
    ddt = dd.from_pandas(df[columns], npartitions=partitions)

    print('table ' + table + ' loaded into dask dataframe')
    return ddt


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


# read nested json
def njson_pd(fname):
    with io.open(fname, encoding='utf-8') as f:
        data = json.load(f)

    data = cx.get_values(data)
    df = pd.DataFrame(data)
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


# from https://stackoverflow.com/questions/32468402/how-to-explode-a-list-inside-a-dataframe-cell-into-separate-rows
@fx.timer
def lateral_explode(df, fieldname):
    df = check_df(df)
    temp_fieldname = fieldname + '_made_tuple_'
    df[temp_fieldname] = df[fieldname].map(tuple)
    list_of_dataframes = []
    temp_list = df[temp_fieldname].unique()
    
    for values in temp_list:
        list_of_dataframes.append(pd.DataFrame({
            temp_fieldname: [values] * len(values),
            fieldname: list(values),
        }))
    df = df[list(set(df.columns) - set([fieldname]))]\
        .merge(pd.concat(list_of_dataframes), how='left', on=temp_fieldname)
    del df[temp_fieldname]

    return df


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


# get a euclidean distance matrix from a pandas field
def get_ecd_mx(df, labels, field):
        # reset index to force index row numbers start from 0
        mx = df[[labels, field]].reset_index(drop=True)

        # build tuples with all possible combinations of indexes
        combinations = [p for p in itertools.product(
            list(range(0, len(mx[field].values))), repeat=2)]

        arr_ = []
        for ix, jx in combinations:
            arr_.append(cx.get_ecd(mx.loc[ix, field], mx.loc[jx, field], square=False))

        arr_ = np.array_split(arr_, len(mx[field].values))
        nx = pd.DataFrame(arr_, columns=df[labels], index=df[labels])

        return nx


def set_df_int(df, exclude):
    '''
    Returns a pandas dataframe converted to integer.

    Parameters:
        exclude(array):The array of columns to exclude from conversion (text e.g.).

    Returns:
        df(dataframe):The pandas dataframe converted to integer.
    '''

    columns = [i for i in df.columns if i not in exclude]

    for col in columns:
        df[col] = pd.to_numeric(df[col])
    
    return df
