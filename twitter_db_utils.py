from python_utils import sqlite_utils as dbx

def insert_tweets(df, db, table):
        if len(df) > 0:
            wildcards = ','.join(['?'] * 16)

            data = []

            for ix in df.to_records(index=False):
                data.append((
                    str(ix['created_at']),
                    str(ix['favorites']),
                    str(ix['hashtags']),
                    str([i.encode('utf-8') for i in ix['links']]),
                    str(ix['mentions']),
                    str(ix['reply']),
                    str(ix['replying']),
                    str(ix['retweets']),
                    str(ix['text'].encode('utf-8')),
                    str(ix['tweet_id']),
                    str(ix['user_id']),
                    str(ix['user_name'].encode('utf-8')),
                    str(ix['user_screen_name']),
                    'NaN',
                    'NaN',
                    'NaN'
                ))

            conn = dbx.connect_db(db)
            c = conn.cursor()
            c.executemany("INSERT OR IGNORE INTO %s VALUES(%s)" % (table, wildcards), data)
            conn.commit()
            conn.close()
        else:
            print('No records to insert.')
