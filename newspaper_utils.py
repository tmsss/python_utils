from newspaper import Article
import newspaper
import os
import json
import pandas as pd
import numpy as np
from python_utils import file_utils as fx
from python_utils import regex_utils as rx
from python_utils import pandas_utils as pdx
from python_utils import requests_utils as rqx
from python_utils import google_utils as gx
from python_utils import sqlite_utils as dbx
from tqdm import tqdm


class ArticleManager(object):
    '''
    Get articles from websites and save them in folder
    '''

    def __init__(self, domain):
        self.df = fx.load_pickle('dataframes/df_links_unique.pkl')
        self.domain = domain
        self.directory = 'articles\\' + self.domain
        self.corpus = 'corpus\\' + self.domain

        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        if not os.path.exists(self.corpus):
            os.makedirs(self.corpus)

    def log_errors(self, error):
        logf = open(self.directory + '/errors.log', 'a')
        logf.write(error + '\n')

    def get_article(self, url, id):
            try:
                a = Article(str(url))
                a.download()
                a.parse()
                doc = {attr: value for attr, value in a.__dict__.items() if not attr.startswith('__') and type(value) in [str, list, set, bool, int, dict, 'collections.defaultdict']}
                fx.save_pickle(self.directory + '/' + str(id) + '.pkl', doc)

            except newspaper.article.ArticleException as e:
                self.log_errors('Error downloading: ' + str(url))
                pass

    def get_archive(self, url, id):
        # From Wayback Machine api docs at https://archive.org/help/wayback_api.php
        archive = rqx.fetch('http://archive.org/wayback/available?url=' + str(url))
        # self.log_errors(str(json.loads(archive.read())))

        try:
            archive = json.loads(archive.read())

            if len(archive['archived_snapshots']) > 0 and archive['archived_snapshots']['closest']['available'] == True:
                self.get_article(archive['archived_snapshots']['closest']['url'], id)
            else:
                self.log_errors('No archive for (else): ' + str(url))
                pass

        except Exception:
            self.log_errors('No archive for (exception): ' + str(url))
            pass

    def filter_df_domain(self):

        df = self.df[self.df['link'].map(lambda x: rx.check_domain(str(x).strip(), self.domain))]

        return df

    def filter_df_id(self, id):
        articles = fx.get_fnames(self.directory)
        df = self.df[self.df[id].isin(articles)]
        return df

    def get_missing_files(self):

        files = fx.get_fnames(self.directory)

        df = self.filter_df_domain()

        df['id'] = df['id'].map(lambda x: str(x).split('.')[0])

        if len(files) > 0:
            return df[~df['id'].isin(files)]
        else:
            return df

    def get_articles(self):

        df = self.get_missing_files()

        for ix, row in tqdm(df.iterrows()):
            self.get_article(str(row['link']).strip(), str(row['id']))

    # get offline urls from the Wayback Machine API
    def get_archives(self):

        df = self.get_missing_files()

        self.log_errors('----------starting archive log----------')

        for ix, row in tqdm(df.iterrows()):
            self.get_archive(str(row['link']).strip(), str(row['id']))

    def set_broken_links(self):

        df = self.get_missing_files()

        df = pdx.check_df(df)

        df['offline'] = True

        pdx.df_update_sql_field('brexit.db', 'links_unique', 'id', 'offline', df, 'BOOLEAN')

    def get_broken_links(self):

        df = self.get_missing_files()

        df['offline'] = True

        for ix, row in tqdm(df.iterrows()):
            row['link'], row['offline'] = gx.get_broken_link(row['link'], rx.find_domain(row['link']))

        df = df[df['offline'] == False]

        if len(df) > 0:
            print('Updating %s records' % len(df))
            pdx.df_update_sql_field('brexit.db', 'links_unique', 'id', 'offline', df, 'BOOLEAN')
            pdx.df_update_sql_field('brexit.db', 'links_unique', 'id', 'link', df, 'BOOLEAN')

    @fx.timer
    def get_corpus(self):

        files = fx.get_fnames(self.directory)

        # remove log file
        files.remove('errors')

        articles = []

        for fname in files:
            path = os.path.join(os.path.abspath(os.curdir) + '\\' + self.directory, fname + '.pkl')
            doc = fx.load_pickle(path)
            article = {
                'id': fname,
                'title': doc['title'],
                'text': doc['text']
                }
            articles.append(article)

        df = pd.DataFrame(articles)
        fx.save_pickle(os.path.join(self.corpus, self.domain + '.pkl'), df)
        pdx.save_to_csv(df, os.path.join(self.corpus, self.domain))


    @fx.timer
    def get_corpus_weight(self, column):
        """
        Get weighted corpus dataframe according to column weight
        (count, favorites, retweets, is_bot)
        """

        df_corpus = fx.load_pickle(os.path.join(self.corpus, self.domain + '.pkl'))
        df_weight = fx.load_pickle(os.path.join(os.path.abspath(os.curdir) + '\\' + 'dataframes', 'top_links_final.pkl'))

        df_weight = pdx.check_df(df_weight)

        df_weight = df_weight.filter(['id', column], axis=1)

        df_corpus['id'] = df_corpus['id'].astype(int)
        df_weight['id'] = df_weight['id'].astype(int)

        df = pd.merge(df_corpus, df_weight, on='id')

        df = pd.DataFrame(np.repeat(df.values, df[column].replace(0,1).tolist(), axis=0), columns=df.columns)

        fx.save_pickle(os.path.join(self.corpus, self.domain + '_' + str(column) + '.pkl'), df)

    def clean_file(self, fname, field, remove, **kwargs):
        media = ('Media', 'Video', 'Image', 'Search', 'Sorry')
        attributes = ['caption', 'copyright', 'playback', 'episode', 'iPlayer', 'radio', 'BBC2']

        doc = fx.load_pickle(self.directory + '\\' + fname)
        lines = doc[field]
        # print(lines)

        if 'split' in kwargs:
            lines = [line for line in lines.split('\n') if not line.startswith(media) or not any(x in line.split() for x in attributes)]
            doc[field] = '\n'.join(lines)

        if 'clean' in kwargs:
            if remove in lines:
                doc[field] = ''
            else:
                pass

        doc[field] = lines.replace(remove, '')
        # print(doc['text'])
        fx.save_pickle(self.directory + '\\' + fname, doc)

    def clean_files(self, field, remove, **kwargs):
        """
        Remove media attributes and irrelevant content
        """
        if 'files_' in kwargs:
            files = kwargs['files_']
        else:
            files = os.listdir(self.directory)
            files.remove('errors.log')

        for fname in tqdm(files):
            self.clean_file(fname, field, remove)


    def clean_directory(self, **kwargs):
        """
        Remove files with with less than 5kb and/or from other domains
        """
        files = fx.get_fnames(self.directory)

        # remove log file
        files.remove('errors')

        files_to_remove = []

        if 'clear_small' in kwargs:
            ids = [f for f in files if os.path.getsize(self.directory + '\\' + f + '.pkl') < 10000]
            files_to_remove.extend([self.directory + '\\' + f + '.pkl' for f in ids])
            dbx.delete_rows('brexit.db', 'links_unique', 'id', ids)

            if len(files_to_remove) > 0:
                fx.delete_files(files_to_remove)

        if 'save' in kwargs:
            data = []
            for id in tqdm(files):
                doc = fx.load_pickle(self.directory + '\\' + id + '.pkl')
                link = doc['canonical_link']
                data.append((id, link))

            df = pd.DataFrame(data, columns=['id', 'link'])
            pdx.save_to_csv(df, os.path.join(self.corpus, self.domain))

    def replace_links(self):
        """
        Find small files, recover original link from google redirect notice
        and update link in DB and crawl links again
        """
        files = fx.get_fnames(self.directory)

        # remove log file
        files.remove('errors')

        ids = []

        ids.extend([f for f in files if os.path.getsize(self.directory + '\\' + f + '.pkl') < 10000])

        links = []
        for id in tqdm(ids):
            doc = fx.load_pickle(self.directory + '\\' + str(id) + '.pkl')
            links.append(rx.get_url(doc['text']))

        data = list(zip(ids, links))

        df = pd.DataFrame(data, columns=['id', 'link'])

        print(df)

        pdx.df_update_sql_field('brexit.db', 'links_unique', 'id', 'link', df, 'TEXT')

        df = df[~df['link'].isnull()]

        for ix, row in df.iterrows():
            self.get_article(row['link'], row['id'])
