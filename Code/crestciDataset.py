import pandas as pd

fake_followers = pd.read_csv('/home/zaimaz/Desktop/research1/BotEvolution/Dataset/cresci-2017.csv/datasets_full.csv/fake_followers.csv/tweets.csv', encoding='ISO-8859-1')
social_spambots = pd.read_csv('/home/zaimaz/Desktop/research1/BotEvolution/Dataset/cresci-2017.csv/datasets_full.csv/social_spambots_3.csv/tweets.csv', encoding='ISO-8859-1')


# print("Column names:")
# print(fake_followers.columns)

# print("\nFirst five rows:")
# print(fake_followers['timestamp'].str[:4].value_counts())

posts_fake_follower = len(fake_followers[fake_followers['in_reply_to_status_id'] == 0]['in_reply_to_status_id'])
posts_social_spambots = len(social_spambots[social_spambots['in_reply_to_status_id'] == 0]['in_reply_to_status_id'])
comments_fake_follower = len(fake_followers[fake_followers['in_reply_to_status_id'] != 0]['in_reply_to_status_id'])
comments_social_spambots = len(social_spambots[social_spambots['in_reply_to_status_id'] != 0]['in_reply_to_status_id'])

print(f'File 1: posts = {posts_fake_follower}, comments = {comments_fake_follower}')
print('File 1 timestamp stats: ')
print("Posts: ")
print(fake_followers[fake_followers['in_reply_to_status_id'] == 0]['timestamp'].str[:4].value_counts())
print("Comments: ")
print(fake_followers[fake_followers['in_reply_to_status_id'] != 0]['timestamp'].str[:4].value_counts())
print(f'File 2: posts = {posts_social_spambots}, comments = {comments_social_spambots}')
print('File 2 timestamp stats: ')
print("Posts: ")
print(social_spambots[social_spambots['in_reply_to_status_id'] == 0]['timestamp'].str[:4].value_counts())
print("Comments: ")
print(social_spambots[social_spambots['in_reply_to_status_id'] != 0]['timestamp'].str[:4].value_counts())
print(f'Total posts: {posts_fake_follower+posts_social_spambots}')
print(f'Total comments: {comments_fake_follower+comments_social_spambots}')









