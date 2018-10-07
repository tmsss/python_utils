import sqlite3
import pandas as pd
import dask
import dask.dataframe as dd


# connect to a db
def connect_db(db):
    conn = sqlite3.connect(db)
    conn.text_factory = str
    # conn.text_factory = bytes
    # conn.text_factory = lambda x: unicode(x, 'utf-8', 'ignore')
    return conn


def get_db_tables(db):
    conn = connect_db(db)
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
    return tables


def update_field(db, table, where, field, type, data):
    conn = connect_db(db)
    c = conn.cursor()
    c.execute('''PRAGMA locking_mode = EXCLUSIVE''')
    c.execute('''PRAGMA synchronous = OFF''')
    c.execute('''PRAGMA journal_mode = OFF''')
    columns = [i[1] for i in c.execute("PRAGMA table_info(" + table + ")")]

    if field not in columns:
        c.execute("ALTER TABLE " + table + " ADD COLUMN " + field + " " + type )

    # c.executemany("UPDATE " + table + " SET " + field + " = ? WHERE " + where + " = ? ", (field_values, where_values))
    c.execute("BEGIN TRANSACTION;")
    c.executemany("UPDATE %s SET %s = ? WHERE %s = ?" % (table, field, where), data)
    c.execute("COMMIT;")
    print('updated %s records in %s' % (len(data), table))
    # conn.commit()


def delete_rows(db, table, where, data):
    conn = connect_db(db)
    c = conn.cursor()
    c.execute('''PRAGMA journal_mode = OFF''')
    c.execute("BEGIN TRANSACTION;")
    c.executemany("DELETE FROM %s WHERE %s = (?)" % (table, where), [(i,) for i in data])
    c.execute("COMMIT;")
    print('deleted %s records from %s' % (len(data), table))
    # conn.commit()


def insert_row(db, table, columns, values):
    conn = connect_db(db)
    c = conn.cursor()

    # fields = []
    # for field, type in zip(columns, types):
    #     fields.append(field + ' ' + type)

    c.execute("CREATE TABLE IF NOT EXISTS %s %s" % (table, columns))
    c.execute("BEGIN TRANSACTION;")
    c.execute("INSERT INTO %s %s VALUES %s;" % (table, columns, values))  # columns and values should be tuples
    c.execute("COMMIT;")
    conn.close()


def db_insert_many(db, table, wildcards, data):
    conn = connect_db(db)
    c = conn.cursor()
    c.execute("BEGIN TRANSACTION;")
    c.executemany("INSERT INTO %s values(%s)" % (table, wildcards), data)
    c.execute("COMMIT;")
    conn.close()


def select_sql_pd_limit(db, table, fields, field, value, limit):
    conn = connect_db(db)

    if type(fields) == list:
        fields = ", ".join(fields)

    value = "'%" + value + "%'"
    query = "SELECT %s FROM %s WHERE %s LIKE %s LIMIT %s;" % (fields, table, field, value, limit)
    df = pd.read_sql_query(query, conn)
    return df


def db_hdf(database, table, chunksize, offset, fname):
    conn = connect_db(database)
    store = pd.HDFStore(fname, mode='a')

    while True:
        query = "SELECT * FROM %s limit %s offset %s;" % (table, chunksize, offset)
        df = pd.read_sql_query(query, conn)
        offset += chunksize
        # df.to_hdf(file, 'tweets', append=True, mode='a')
        # store.append('tweets', df, min_itemsize={ 'values_block_1' : 2500 })
        store.append('tweets', df, data_columns=df.columns, min_itemsize=2000, index=False)

        if df.shape[0] < chunksize:
            store.close()
            break


def query_sql_pd(db, table, fields, order, direction, limit):
    conn = connect_db(db)

    query = "SELECT %s FROM %s ORDER BY %s %s LIMIT %s;" % (fields, table, order, direction, limit)

    df = pd.read_sql_query(query, conn)

    return df


def count_sql_df(db, table, fields, field):
    conn = connect_db(db)

    if type(fields) == list:
        fields = ", ".join(fields)

    query = "SELECT %s, count(*) FROM %s GROUP BY %s;" % (fields, table, field)

    df = pd.read_sql_query(query, conn)

    return df


def count_having_sql_df(db, table, fields, groupby, field, operator, value):
    conn = connect_db(db)

    if type(fields) == list:
        fields = ", ".join(fields)

    query = "SELECT %s, count(*) FROM %s GROUP BY %s HAVING %s%s'%s';" % (fields, table, groupby, field, operator, value)

    df = pd.read_sql_query(query, conn)
    return df


def db_ddf_slice(db, table, columns, partitions, chunksize, offset, slice):
    conn = connect_db(db)
    df = pd.DataFrame()

    while True:
        query = "SELECT * FROM %s limit %s offset %s;" % (table, chunksize, offset)
        df = pd.read_sql_query(query, conn)
        ddt = dd.from_pandas(df[columns], npartitions=partitions)
        if offset == 0:
            final = ddt
        else:
            final = dd.concat([ddt, final], axis=0, interleave_partitions=True)
            if len(final) >= slice:
                break

        offset += chunksize

        if df.shape[0] < chunksize:
            break

    print('table ' + table + ' loaded into dask dataframe')
    return final
