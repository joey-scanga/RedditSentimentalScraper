import praw, time, sys, re
from dotenv import dotenv_values
from praw.models import MoreComments
import pandas as pd
from transformers import pipeline

config = dotenv_values(".env")

reddit = praw.Reddit(client_id=config["CLIENT_ID"],
                     client_secret=config["CLIENT_SECRET"],
                     user_agent=config["USER_AGENT"])

'''
Get a list of hot posts from a given subreddit, returns a list of objects as in getCommentsFromOnePost().
'''
def getListOfPostsFromSubreddit(subName, limit=1):
    if limit < 1:
        print(f"Invalid number of posts (limit={limit})")
        return
    posts = []
    sub = reddit.subreddit(subName)
    i = 1
    for post in sub.hot(limit=limit):
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

'''
Get a specific post.
'''
def getCommentsFromOnePost(postID):
    post = {"myId": postID, "comments": [], "nComments": 0}
    submission = reddit.submission(postID)
    if not submission:
        print(f"Invalid postID (id: {postID})")
    for top_level_comment in submission.comments:
        if isinstance(top_level_comment, MoreComments):
            continue
        post["comments"].append(top_level_comment.body[:512]) #Token limit is 512 characters
        post["nComments"] += 1
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
starttime = time.time()
url = input("Enter a Reddit URL: ")
postid = re.findall("\/comments\/\w+\/", url)[0][10:-1]
post = getCommentsFromOnePost(postid)
if not post:
    sys.exit(-1)
print("\nNumber of comments: ", post["nComments"], "\n")
print("Analysis start\n")

result = getAvg(getSentiment(post["comments"]))

print(f"Positive: {result[0]}")
print(f"Neutral: {result[1]}")
print(f"Negative: {result[2]}")
endtime = time.time()
print("Running time: ", format(endtime - starttime, '.2f'), " seconds", sep="")


