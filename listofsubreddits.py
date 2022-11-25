import praw
from praw.models import MoreComments
import pandas as pd
from transformers import pipeline

reddit = praw.Reddit(client_id='X7NE499Moo8VQhudVe6zmg',
                     client_secret='8LWwUA7UNCHAfcFJ_pNnWdeC-Rf68Q',
                     user_agent='comment-scraper')

'''
Get a specific post.
'''
# submission = reddit.submission("z40utz")
# comments = []
# for top_level_comment in submission.comments:
#     if isinstance(top_level_comment, MoreComments):
#         continue
#     print(top_level_comment.body, "\n")

'''
Get a list of hot posts from a given subreddit.
'''
getListOfPostsFromSubreddit(subName):
    posts = []
    sub = reddit.subreddit(subName)
    i = 1
    for post in sub.hot(limit=1):
        print(f"Iteration: {i}")
        postobj = {"myid": post.id, "comments": []}
        sub = reddit.submission(post.id)
        j = 20
        for top_level_comment in sub.comments:
            if j > 20:
                break
            if isinstance(top_level_comment, MoreComments):
                continue
            postobj["comments"].append(top_level_comment.body)
        posts.append(postobj)
        print(f"Iteration: {i} complete")
        i += 1
    return posts

print("Analysis start\n")

def getSentiment(data):
    sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert", top_k=None)
    output = sentiment_pipeline(data)
    return output

def getAvg(sentimentdata):
    positive = 0
    negative = 0
    neutral = 0
    postNum = 0
    for sentiment in sentimentdata:
        postNum += 1
        for classification in sentiment:
            if classification["label"] == "positive":
                positive += classification["score"]
            elif classification["label"] == "negative":
                negative += classification["score"]
            elif classification["label"] == "neutral":
                neutral += classification["score"]
    return [positive/postNum, neutral/postNum, negative/postNum]

posts = getListOfPostsFromSubreddit("WallStreetBets")

for post in posts:
    result = getAvg(getSentiment(post["comments"]))
    print(f"Positive: {result[0]}")
    print(f"Neutral: {result[1]}")
    print(f"Negative: {result[2]}")


