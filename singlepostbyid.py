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
def getCommentsFromOnePost(postID):
    post = {"myId": postID, "comments": []}
    submission = reddit.submission(postID)
    for top_level_comment in submission.comments:
        if isinstance(top_level_comment, MoreComments):
            continue
        post["comments"].append(top_level_comment.body)
    return post


def getSentiment(data):
    sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert", top_k=None, device=0)
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

'''
Testing
'''
print("Analysis start\n")
post = getCommentsFromOnePost("z4eth1")

result = getAvg(getSentiment(post["comments"]))

print(f"Positive: {result[0]}")
print(f"Neutral: {result[1]}")
print(f"Negative: {result[2]}")


