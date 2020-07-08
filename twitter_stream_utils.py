from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import time
import csv
import sys


# Create a streamer object
class StdOutListener(StreamListener):

    # Define a function that is initialized when the miner is called
    def __init__(self, api = None):
        # That sets the api
        self.api = api
        # Create a file with 'data_' and the current time
        self.filename = 'data'+'_'+ time.strftime('%Y%m%d-%H%M%S') +'.csv'
        # Create a new file with that filename
        csvFile = open(self.filename, 'w')

        # Create a csv writer
        csvWriter = csv.writer(csvFile)

        # Write a single row with the headers of the columns
        csvWriter.writerow(['text',
                            'created_at',
                            'geo',
                            'lang',
                            'place',
                            'coordinates',
                            'user.favourites_count',
                            'user.statuses_count',
                            'user.description',
                            'user.location',
                            'user.id',
                            'user.created_at',
                            'user.verified',
                            'user.following',
                            'user.url',
                            'user.listed_count',
                            'user.followers_count',
                            'user.default_profile_image',
                            'user.utc_offset',
                            'user.friends_count',
                            'user.default_profile',
                            'user.name',
                            'user.lang',
                            'user.screen_name',
                            'user.geo_enabled',
                            'user.profile_background_color',
                            'user.profile_image_url',
                            'user.time_zone',
                            'id',
                            'favorite_count',
                            'retweeted',
                            'source',
                            'favorited',
                            'retweet_count'])

    # When a tweet appears
    def on_status(self, status):

        # Open the csv file created previously
        csvFile = open(self.filename, 'a')

        # Create a csv writer
        csvWriter = csv.writer(csvFile)

        # If the tweet is not a retweet
        if not 'RT @' in status.text:
            # Try to
            try:
                # Write the tweet's information to the csv file
                csvWriter.writerow([status.text,
                                    status.created_at,
                                    status.geo,
                                    status.lang,
                                    status.place,
                                    status.coordinates,
                                    status.user.favourites_count,
                                    status.user.statuses_count,
                                    status.user.description,
                                    status.user.location,
                                    status.user.id,
                                    status.user.created_at,
                                    status.user.verified,
                                    status.user.following,
                                    status.user.url,
                                    status.user.listed_count,
                                    status.user.followers_count,
                                    status.user.default_profile_image,
                                    status.user.utc_offset,
                                    status.user.friends_count,
                                    status.user.default_profile,
                                    status.user.name,
                                    status.user.lang,
                                    status.user.screen_name,
                                    status.user.geo_enabled,
                                    status.user.profile_background_color,
                                    status.user.profile_image_url,
                                    status.user.time_zone,
                                    status.id,
                                    status.favorite_count,
                                    status.retweeted,
                                    status.source,
                                    status.favorited,
                                    status.retweet_count])
            # If some error occurs
            except Exception as e:
                # Print the error
                print(e)
                # and continue
                pass

        # Close the csv file
        csvFile.close()

        # Return nothing
        return

    # When an error occurs
    def on_error(self, status_code):
        # Print the error code
        print('Encountered error with status code:', status_code)

        # If the error code is 401, which is the error for bad credentials
        if status_code == 401:
            # End the stream
            return False

    # When a deleted tweet appears
    def on_delete(self, status_id, user_id):

        # Print message
        print("Delete notice")

        # Return nothing
        return

    # When reach the rate limit
    def on_limit(self, track):

        # Print rate limiting error
        print("Rate limited, continuing")

        # Continue mining tweets
        return True

    # When timed out
    def on_timeout(self):

        # Print timeout message
        print(sys.stderr, 'Timeout...')

        # Wait 10 seconds
        time.sleep(10)

        # Return nothing
        return

def start_mining(keywords):
    #Variables that contains the user credentials to access Twitter API

    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    stream = Stream(auth, l)

    stream.filter(track = keywords)


# start_mining(['#eleicaobrasil', '#eleiçãobrasil'])

#import json
#from tweepy import Cursor
#from twitter_client import get_twitter_client
#
#
#if __name__ == '__main__':
#  client = get_twitter_client()
#
#  with open('home_timeline.jsonl', 'w') as f:
#      for page in Cursor(client.home_timeline, count=200).pages(4):
#          for status in page:
#              f.write(json.dumps(status._json)+"\n")

#import string
#import time
#from tweepy import Stream
#from tweepy.streaming import StreamListener
#from twitter_client import get_twitter_auth
#
#class CustomListener(StreamListener):
#        def __init__(self, fname):
#            safe_fname = format_filename(fname)
#            self.outfile ="stream_%s.jsonl" %safe_fname
#
#        def on_data(self,data):
#            try:
#                with open(self.outfile, 'a') as f:
#                    f.write(data)
#                    return True
#            except BaseException as e:
#                    sys.stderr.write("Error on_data: {}\n".format(e))
#                    time.sleep(5)
#            return True
#
#        def on_error(selg, status):
#            if status == 420:
#                sys.stderr.write("Rate limit exceeded\n")
#                return False
#            else:
#                sys.stderr.write("Error {}\n".format(status))
#                return True
#
#def format_filename(fname):
#    return ' '.join(convert_valid(one_char) for one_char in fname)
#
#def convert_valid(one_char):
#    valid_chars ="-_.%s%s" % (string.ascii_letters, string.digits)
#    if one_char in valid_chars:
#        return one_char
#    else:
#        return '_'
#
#if __name__ == '__main__':
#    query = sys.argv[1:]
#    query_fname = ' '.join(query)
#    auth = get_twitter_auth()
#    twitter_stream = Stream(auth, CustomListener(query_fname))
#    twitter_stream.filter(track=query, async=True)
