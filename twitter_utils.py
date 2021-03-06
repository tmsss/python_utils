import time
from tweepy import API
from tweepy import OAuthHandler
from tweepy import TweepError
from tweepy import parsers
import file_utils as fx
import date_utils as dx
import apis


class TwitterManager(object):

    def __init__(self, **kwargs):
        """Setup Twitter authentication.
        Return: tweepy.OAuthHandler object
        """

        api = apis.twitter_api()
        auth = OAuthHandler(api['consumer_key'], api['consumer_secret'])
        auth.set_access_token(api['access_token'], api['access_secret'])

        self.api = API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, parser=parsers.JSONParser())
        self.initial = True
        self.search = True
        self.max_id = ''
        # self.since_id = ''
        self.attributes = kwargs

    def get_tweet_id(self, query, count, date):
        ''' Function that gets the ID of a tweet. This ID can then be
        used as a 'starting point' from which to search. The query is
        required and has been set to a commonly used word by default.
        The variable 'days_ago' has been initialized to the maximum
        amount we are able to search back in time (9).'''

        tweet = self.get_query(query, count, until=date)

        if len(tweet['statuses']) > 0:
            return tweet['statuses'][0]['id']
        else:
            return False

    def get_query(self, query, max_tweets, **kwargs):
        ''' Function that takes in a search string 'query', the maximum
            number of tweets 'max_tweets', and the minimum (i.e., starting)
            tweet id. It returns a list of tweepy.models.Status objects. '''

        return self.api.search(q=query, count=max_tweets, **kwargs)

    def get_search_page(self, query, max_tweets, since_id, max_id):
        # inspired in https://galeascience.wordpress.com/2016/03/18/collecting-twitter-data-with-python/

        searched_tweets = []

        while len(searched_tweets) < max_tweets:
            remaining_tweets = max_tweets - len(searched_tweets)
            try:
                tweets = self.get_query(query, remaining_tweets, since_id=str(since_id), max_id=str(max_id-1))

                print('found', len(tweets['statuses']), 'tweets')

                if len(tweets['statuses']) == 0:
                    print('no tweets found')
                    self.search = False
                    self.initial = True
                    break

                searched_tweets.extend(tweets['statuses'])
                self.max_id = tweets['statuses'][-1]['id']

                return searched_tweets

            except TweepError as e:
                print(e)
                print('exception raised, waiting 15 minutes')
                time.sleep(15*60)
                break

    def get_search_tweets(self, keywords, max_tweets, days_ago):
        ''' This is a script that continuously searches for tweets
        that were created over a given number of days. The search
        dates and search phrase can be changed below.

        runtime limit in hours
        number of tweets per search (will be iterated over) - maximum is 100
        search limits e.g., from 7 to 8 gives current weekday from last week,
        min_days_old=0 will search from right now

        this geocode includes nearly all American states (and a large portion of Canada)
        USA = '39.8,-95.583068847656,2500km'
       '''

        # loop over search items,
        # creating a new file for each
        for query in keywords:

            self.search = True

            print('Search phrase =', query)
            fx.create_folder('data/' + query)

            start = dx.get_date(0, date_object=True)
            end = dx.get_date(days_ago, date_object=True)

            range = dx.date_range_day(start, end, 1)

            for day in range:

                tweets_tosave = []

                while self.search is True:

                    if self.initial is True:
                        start = dx.get_date(1)
                        end = day
                        self.max_id = self.get_tweet_id(query, 1, end)
                        since_id = self.get_tweet_id(query, 1, start)
                        self.initial = False
                        if since_id is False:
                            pass

                    tweets = self.get_search_page(query, max_tweets, since_id, self.max_id)

                    if tweets:
                        tweets_tosave.extend(tweets)

                        if len(tweets_tosave) >= 1000:
                            fname = dx.format_date_str(tweets_tosave[-1]['created_at'], '%a %b %d %H:%M:%S %z %Y', '%d-%m-%y_%H-%M-%S') + '_' + str(tweets_tosave[-1]['id'])
                            fx.save_json('data/' + query + '/' + str(fname), tweets_tosave)
                            tweets_tosave = []
                    else:
                        pass

    def get_user_details(self, username):
        try:
            print('user id: ' + str(username))
            userobj = self.api.get_user(username)
            return userobj
        except TweepError as e:
            print(e)
            # 63: user suspended
            # 50: user not found
            if e.api_code == 63 or e.api_code == 50:
                return e

    def get_user_timeline(self, username):
        try:
            print('reading: ' + str(username))
            status = self.api.user_timeline(username)
            return status
        except TweepError as e:
            if e.api_code == 63 or e.api_code == 50:
                return True

            print('error: ' + str(e))
            return False
