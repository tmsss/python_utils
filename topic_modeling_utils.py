from nltk.corpus import stopwords
import numpy as np
import re
import os
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag, stanford
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import bokeh.plotting as bp
from bokeh.plotting import save
from bokeh.models import HoverTool
from python_utils import pandas_utils as pdx
from python_utils import file_utils as fx
from python_utils import regex_utils as rx

# configure java home variable for NERTagger
os.environ['JAVAHOME'] = "C:\\Program Files\\Java\\jdk1.8.0_101\\bin\\java.exe"


class build_model(object):

    def __init__(self, media, label, model, vectorizer, df, topics, field, delete, **kwargs):
        self.media = media
        self.label = label
        self.model = model
        self.vectorizer = vectorizer
        self.df = pdx.check_df(df)
        self.topics = topics
        self.field = field
        self.path = 'models/' + self.label + '/' + self.label
        self.delete = delete
        self.attributes = kwargs

        if not os.path.exists('models/' + self.label):
            os.makedirs('models/' + self.label)

        if self.delete:
            fx.clean_folder('models/' + self.label)

    def load_data(self):

        if 'cross_val' in self.attributes:
            train, test = self.df.random_split([0.80, 0.20])
            print('Documentos no conjunto de treino: %d' % (len(train)))
            print('Documentos no conjunto de teste: %d' % (len(test)))
        else:
            train = self.df
            test = []

        return train, test

    def clean_data(self, text):

        if 'strip_urls' in self.attributes:
            text = rx.strip_urls(text)

        stop_words = stopwords.words('english')

        stop_words.extend(stopwords.words('french'))

        stop_words.extend(stopwords.words('spanish'))

        stop_words.extend(stopwords.words('german'))

        stop_words.extend(['brexit', 'twitter', 'tweet', 'euref', 'eureferendum', 'correspondent',
                            'referendum', 'pic', 'eurefpic', 'eupic', 'com', 'bbc'
                            'co', 'html', 'tweet', 'página', 'anterior', 'iplayer', 'la',
                            'pretender', 'pode', 'episode', 'http', 'www', 'javascript',
                            'que', 'pic', 'de', 'android', 'source', 'medium', 'video', 'mr',
                            'bloomerg', 'economist', self.media])

        # remove stop words
        text = [word for word in text.split() if word.lower() not in stop_words]

        if 'stemming' in self.attributes:
            tagger = PorterStemmer()
            text = [tagger.stem(w) for w in text]

        if 'lemmatization' in self.attributes:
            wordnet_lemmatizer = WordNetLemmatizer()
            text = [wordnet_lemmatizer.lemmatize(w, pos='v') for w in text]

        # retrieve only nouns
        if 'pos_tag' in self.attributes:
            tagged = pos_tag(text)
            text = [word for word, pos in tagged if re.findall(r'NN', pos)]

        if 'ner' in self.attributes:
            path = os.path.abspath(os.curdir) + '\\utils\\stanford-ner-2018-02-27\\'
            tagger = stanford.StanfordNERTagger(path + 'classifiers\\english.all.3class.distsim.crf.ser.gz',
                     path + 'stanford-ner.jar')

            text = tagger.tag(text)
            text = [word + '_' + entity for word, entity in text]

        # print(" ".join(text))
        return " ".join(text)

    @fx.timer
    def transform(self):
        train, test = self.load_data()
        train[self.field] = train[self.field].map(lambda x: self.clean_data(x))
        docs = train[self.field]

        vectors = self.vectorizer.fit_transform(docs)
        model_fit = self.model.fit_transform(vectors)
        vectorizer = self.vectorizer

        # save the docs, model, vectorizer and fitted data to disk
        joblib.dump(docs, self.path + '_docs.pkl')
        joblib.dump(self.model, self.path + '.pkl')
        joblib.dump(vectors, self.path + '_vectors.pkl')
        joblib.dump(model_fit, self.path + '_fit.pkl')
        joblib.dump(vectorizer, self.path + '_vectorizer.pkl')

    def check_model(self):
        fname_docs = self.path + '_docs.pkl'
        fname_model = self.path + '.pkl'
        fname_vectors = self.path + '_vectors.pkl'
        fname_fit = self.path + '_fit.pkl'
        fname_vectorizer = self.path + '_vectorizer.pkl'

        if os.path.isfile(fname_model) is False:
            self.transform()
        else:
            print('Files found. No transformation applied.')

        docs = joblib.load(fname_docs)
        model = joblib.load(fname_model)
        vectors = joblib.load(fname_vectors)
        fit = joblib.load(fname_fit)
        vectorizer = joblib.load(fname_vectorizer)

        return docs, model, vectors, fit, vectorizer

    def model_info(self):

        docs, model, vectors, fit, vectorizer = self.check_model()

        keywords = np.array(vectorizer.get_feature_names())

        print("Model: " + self.label)

        if self.label == 'Latent Dirichlet Allocation':
            # Log Likelyhood: Higher the better
            print("Log Likelihood: ", model.score(vectors))

            # Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
            print("Perplexity: ", model.perplexity(vectors))
            print("-" * 80)

        # See model parameters
        self.print_topics()
        text = self.join_topics()
        print("-" * 80)
        print('Vocabulary lenght: %s' % (len(keywords)))
        self.show_wordcloud(text)

    def topic_df(self, n_words=10):

        docs, model, vectors, fit, vectorizer = self.check_model()

        keywords = np.array(vectorizer.get_feature_names())
        topic_keywords = []

        for topic_weights in model.components_:
            top_keyword_locs = (-topic_weights).argsort()[:n_words]
            topic_keywords.append(keywords.take(top_keyword_locs))

        df = pd.DataFrame(topic_keywords)
        df.columns = ['Word ' + str(i) for i in range(df.shape[1])]
        df.index = ['Topic ' + str(i) for i in range(df.shape[0])]

        return df

    def topic_keywords_df(self, n_topics=True):

        docs, model, vectors, fit, vectorizer = self.check_model()

        # create topic-keyword matrix from model
        df = pd.DataFrame(model.components_)

        # assign column and index names
        df.columns = vectorizer.get_feature_names()
        df.index = ["Topic " + str(ix) for ix in range(model.n_components)]

        # include only the number of top topics specified in self.topics in the dataframe
        if n_topics:
            keywords = []
            for idx, topic in enumerate(model.components_[:self.topics]):
                keywords.extend([vectorizer.get_feature_names()[i] for i in topic.argsort()[:-self.topics - 1:-1]])

            df = df[keywords]

        return df

    def print_topics(self):

        docs, model, vectors, fit, vectorizer = self.check_model()

        for idx, topic in enumerate(model.components_[:self.topics]):
            print("Topic %d: " % (idx + 1) + " ".join([vectorizer.get_feature_names()[i] for i in topic.argsort()[:-self.topics - 1:-1]]))


    def find_topics(self, original_doc, new_doc):

        topics = []
        counter = 0

        for i in original_doc:
            if i in new_doc.split():
                topics.append(i)
                counter += 1
        print('Found ' + str(counter) + ' topics: ' + ", ".join(topics))


    # join topics
    def join_topics(self):

        docs, model, vectors, fit, vectorizer = self.check_model()

        text = ""
        for idx, topic in enumerate(model.components_[:self.topics]):
            text += " ".join([vectorizer.get_feature_names()[i] for i in topic.argsort()[:-self.topics - 1:-1]])
        return text

    # display word cloud
    def show_wordcloud(self, text):
        wordcloud = WordCloud().generate(text)
        fig = plt.figure()
        fig.set_figwidth(14)
        fig.set_figheight(18)
        plt.imshow(wordcloud)
        plt.title(self.label, size=30, y=1.01)
        plt.axis('off')
        plt.show()

    def get_topic_weights(self):
        docs, model, vectors, fit, vectorizer = self.check_model()
        topic_vectors = []

        for idx, topic in enumerate(model.components_[:self.topics]):
            topic_vectors.append(topic.argsort()[:-10 - 1:-1])
    
        return topic_vectors

    def get_topic_dist_avg(self):
        """
        get measure to calculate distance between corpus https://stats.stackexchange.com/questions/102932/comparing-topic-distributions-between-corpora-using-latent-dirichlet-allocation/102981#102981?newreg=785582e9497e44b6b317bc3f098cfb3d
        """
        return np.average(np.array(self.get_topic_weights()), axis=0)

    # performing similarity queries
    def most_similar(self, x, Z, top_n=5):
        dists = euclidean_distances(x.reshape(1, -1), Z)
        pairs = enumerate(dists[0])
        most_similar = sorted(pairs, key=lambda item: item[1])[:top_n]
        return most_similar

    # find similar documents
    def find_similar(self):
        train, test = self.load_data()
        docs, model, vectors, fit, vectorizer = self.check_model()

        print('Most similar documents for the ' + self.label + ' model:')

        for ix, text in enumerate(test):
            x = self.model.transform(self.vectorizer.transform([text]))[0]
            similarities = self.most_similar(x, model)
            document_id, similarity = similarities[0]
            print('original: ' + text + ' [...]')
            similar = self.df.text.iloc[document_id]
            print('similar: ' + similar + ' [...]')
            self.find_topics(self.remove_stopwords(text), self.remove_stopwords(similar))
            print("-" * 80)

    # Infer topics for new documents
    @fx.timer
    def infer(self):

        # adicionar condicional para verificar se cross_val
        train, test = self.load_data()

        self.check_model()

        print('Inference for the ' + self.label + ' model:')

        for txt, id_ in zip(test['text'], test['tweet_id']):
            print('Original: ' + txt[:100] + '...')
            topic, scores = self.predict_topics(txt)
            print('Topic predicition: ' + ', '.join(topic))
            print("-" * 80)

    def predict_topics(self, text):

        docs, model, vectors, fit, vectorizer = self.check_model()

        df = self.topic_df()
        vectorized = vectorizer.transform([self.clean_data(text)])
        topic_probability_scores = model.transform(vectorized)
        topic = df.iloc[np.argmax(topic_probability_scores), :].values.tolist()

        return topic, topic_probability_scores

    @fx.timer
    def best_lda_model(self):

        docs, model, vectors, fit, vectorizer = self.check_model()

        # Define Search Param
        search_params = {'n_components': [10, 15, 20, 25, 30], 'learning_decay': [.5, .7, .9, 1]}

        # Init the Model
        lda = LatentDirichletAllocation(learning_method='online', random_state=5)

        # Init Grid Search Class
        grid_ = GridSearchCV(lda, param_grid=search_params)

        # Do the Grid Search
        grid_.fit(vectors)

        # Best Model
        best_lda_model = grid_.best_estimator_

        # Model Parameters
        print("Best Model's Params: ", grid_.best_params_)

        # Log Likelihood Score
        print("Best Log Likelihood Score: ", grid_.best_score_)

        # Perplexity
        print("Model Perplexity: ", best_lda_model.perplexity(vectors))

        # Get Log Likelyhoods from Grid Search Output
        n_topics = [10, 15, 20, 25, 30]
        decay = [.5, .7, .9, 1]

        # Show graph
        plt.figure(figsize=(12, 8))

        for ix in decay:
            var_name = "log_likelyhoods_{0}".format(ix)
            var_name = [round(gscore.mean_validation_score) for gscore in grid_.grid_scores_ if gscore.parameters['learning_decay']==ix]
            plt.plot(n_topics, var_name, label=str(ix))

        plt.title("Choosing Optimal LDA Model")
        plt.xlabel("Num Topics")
        plt.ylabel("Log Likelyhood Scores")
        plt.legend(title='Learning decay', loc='best')
        plt.show()

    def cluster_documents(self):

        docs, model, vectors, fit, vectorizer = self.check_model()

        # Construct the k-means clusters
        clusters = KMeans(n_clusters=10, random_state=100).fit_predict(fit)

        # Build the Singular Value Decomposition(SVD) model
        svd_model = TruncatedSVD(n_components=2)  # 2 components
        lda_output_svd = svd_model.fit_transform(fit)

        # X and Y axes of the plot using SVD decomposition
        x = lda_output_svd[:, 0]
        y = lda_output_svd[:, 1]

        # Weights for the 15 columns of lda_output, for each component
        print("Component's weights: \n", np.round(svd_model.components_, 2))

        # Percentage of total information in 'lda_output' explained by the two components
        print("Perc of Variance Explained: \n", np.round(svd_model.explained_variance_ratio_, 2))

        # Plot
        plt.figure(figsize=(12, 12))
        plt.scatter(x, y, c=clusters)
        plt.xlabel('Component 2')
        plt.xlabel('Component 1')
        plt.title("Segregation of Topic Clusters", )
        plt.show()

    def tsne(self):

        docs, model, vectors, fit, vectorizer = self.check_model()

        # a t-SNE model
        # angle value close to 1 means sacrificing accuracy for speed
        # pca initializtion usually leads to better results
        fname_tsne = 'models/' + self.label + '_tsne.pkl'

        if os.path.isfile(fname_tsne) is False:
            tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
            tsne_lda = tsne_model.fit_transform(fit)
            joblib.dump(tsne_lda, fname_tsne)
        else:
            tsne_lda = joblib.load(fname_tsne)

        # threshold = .5

        # _idx = np.amax(fit, axis=1) > threshold  # idx of news that > threshold
        # _topics = fit[_idx]
        _topics = fit

        num_example = len(_topics)

        n_top_words = 5  # number of keywords we show

        # 20 colors
        colormap = np.array([
            "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
            "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
            "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
            "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"
        ])

        _lda_keys = []
        for i in range(_topics.shape[0]):
            _lda_keys += _topics[i].argmax(),

        topic_summaries = []
        topic_word = model.components_  # all topic words
        vocab = vectorizer.get_feature_names()

        for i, topic_dist in enumerate(topic_word):
            topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]  # get!
            topic_summaries.append(' '.join(topic_words))  # append!

        # plot_dict = dict(zip(tsne_lda[:, 0], tsne_lda[:, 1], colormap[_lda_keys][:num_example], docs[:num_example].tolist(), _lda_keys[:num_example]))

        plot_dict = {
            'x': tsne_lda[:num_example, 0],
            'y': tsne_lda[:num_example, 1],
            'colors': colormap[_lda_keys][:num_example],
            'content': docs[:num_example].tolist(),
            'topic_key': _lda_keys[:num_example]
        }

        plot_df = pd.DataFrame.from_dict(plot_dict)

        source = bp.ColumnDataSource(data=plot_df)

        title = 'LDA viz'

        plot_lda = bp.figure(plot_width=1400, plot_height=1100,
                             title=title,
                             tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
                             x_axis_type=None, y_axis_type=None, min_border=1)

        plot_lda.scatter('x', 'y', color='colors', source=source)

        '''randomly choose a news (within a topic) coordinate
        as the crucial words coordinate '''
        topic_coord = np.empty((fit.shape[1], 2)) * np.nan
        for topic_num in _lda_keys:
            if not np.isnan(topic_coord).any():
                break
            topic_coord[topic_num] = tsne_lda[_lda_keys.index(topic_num)]

        # plot crucial words
        for i in range(fit.shape[1]):
            plot_lda.text(topic_coord[i, 0], topic_coord[i, 1], [topic_summaries[i]])

        # hover tools
        hover = plot_lda.select(dict(type=HoverTool))
        hover.tooltips = {"content": "@content - topic: @topic_key"}

        bp.output_file('{}.html'.format(title))

        # save the plot
        save(plot_lda)

    def get_graph(self):

        df = self.topic_keywords_df()

        # df.reset_index(drop=True, inplace=True)
        #
        # mx = df.T.dot(df)
        # np.fill_diagonal(mx.values, 0)
        #
        # mx.sum(axis=1).order(ascending=False).head(10)
        #
        print(df)
