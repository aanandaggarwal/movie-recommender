import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from surprise import SVD, Dataset, Reader
from surprise.model_selection import KFold as SurpriseKFold
from surprise import accuracy
from collections import defaultdict

# Load datasets
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')
tags = pd.read_csv('tags.csv')
links = pd.read_csv('links.csv')

# Split ratings into training and testing sets
train_ratings, test_ratings = train_test_split(ratings, test_size=0.2, random_state=42)

# Save training and testing sets
train_ratings.to_csv('train_ratings.csv', index=False)
test_ratings.to_csv('test_ratings.csv', index=False)

print("Training and testing datasets created.")

# Load dataset into Surprise library
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Cross-validation using KFold
kf = SurpriseKFold(n_splits=5, random_state=42)
mae_scores = []
rmse_scores = []

for trainset, testset in kf.split(data):
    model = SVD()
    model.fit(trainset)
    predictions = model.test(testset)
    
    # Calculate MAE and RMSE
    mae_scores.append(accuracy.mae(predictions, verbose=False))
    rmse_scores.append(accuracy.rmse(predictions, verbose=False))

print(f"Average MAE: {np.mean(mae_scores)}")
print(f"Average RMSE: {np.mean(rmse_scores)}")

# Get Top-N recommendations for each user
def get_top_n(predictions, n=10):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    
    return top_n

# Generate Top-10 recommendations
top_n = get_top_n(predictions, n=10)

# Calculate evaluation metrics
def calculate_metrics(top_n, testset):
    hits, total_relevant, total_recommended = 0, 0, 0
    ndcg_sum = 0

    test_dict = defaultdict(set)
    for user_id, movie_id, rating in testset:
        if rating >= 3.0:
            test_dict[user_id].add(movie_id)

    for user_id in top_n:
        recommended_movies = set(iid for (iid, _) in top_n[user_id])
        relevant_movies = test_dict[user_id]

        hits += len(recommended_movies & relevant_movies)
        total_recommended += len(recommended_movies)
        total_relevant += len(relevant_movies)

        # Calculate NDCG
        user_ndcg = 0
        ideal_dcg = 0
        for rank, (iid, _) in enumerate(top_n[user_id], start=1):
            if iid in relevant_movies:
                user_ndcg += 1 / np.log2(rank + 1)
        if len(relevant_movies) > 0:
            ideal_dcg = sum([1 / np.log2(i + 1) for i in range(1, len(relevant_movies) + 1)])
        if ideal_dcg > 0:
            ndcg_sum += user_ndcg / ideal_dcg

    precision = hits / total_recommended if total_recommended > 0 else 0
    recall = hits / total_relevant if total_relevant > 0 else 0
    f_measure = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0
    ndcg = ndcg_sum / len(top_n)

    return precision, recall, f_measure, ndcg

precision, recall, f_measure, ndcg = calculate_metrics(top_n, testset)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F-measure: {f_measure}")
print(f"NDCG: {ndcg}")

# Adding explainability to recommendations
def explain_recommendations(top_n, movies_df, tags_df):
    explanations = {}
    for user_id, recommendations in top_n.items():
        user_explanations = []
        for movie_id, _ in recommendations:
            movie_info = movies_df[movies_df['movieId'] == movie_id]
            user_tags = tags_df[(tags_df['userId'] == user_id) & (tags_df['movieId'] == movie_id)]
            if not movie_info.empty:
                title = movie_info['title'].values[0]
                genres = movie_info['genres'].values[0]
                explanation = f"Recommended because you have shown interest in similar genres: {genres}."
                if not user_tags.empty:
                    tags = ', '.join(user_tags['tag'].values)
                    explanation += f" Tags you used for similar movies: {tags}."
                user_explanations.append((title, explanation))
        explanations[user_id] = user_explanations
    return explanations

# Generate explanations for Top-10 recommendations
recommendation_explanations = explain_recommendations(top_n, movies, tags)

# Print explanations for the first few users
for user_id in list(recommendation_explanations.keys())[:5]:
    print(f"User {user_id}:")
    for title, explanation in recommendation_explanations[user_id]:
        print(f"  Movie: {title}, Explanation: {explanation}")
    print()
