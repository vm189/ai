# Developing a recommendation system using collaborative filtering.

import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Create a sample dataset
data = {
    'user_id': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
    'item_id': [101, 102, 103, 101, 103, 104, 101, 102, 104, 102, 103, 104, 101, 103, 104],
    'rating': [5, 3, 4, 4, 5, 2, 2, 5, 4, 3, 4, 5, 5, 3, 4]
}

df = pd.DataFrame(data)

# Define a Reader object
reader = Reader(rating_scale=(1, 5))  # Adjust the rating scale according to your dataset

# Load the dataset into Surprise
data = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)

# Split the dataset into training and testing sets
trainset, testset = train_test_split(data, test_size=0.2)

# Create an SVD model
model = SVD()

# Train the model on the training set
model.fit(trainset)

# Make predictions on the test set
predictions = model.test(testset)

# Calculate and print RMSE
rmse = accuracy.rmse(predictions)
print(f'RMSE: {rmse}')

def get_top_n_recommendations(predictions, n=10):
    # Convert the predictions into a DataFrame
    top_n = {}
    for uid, iid, true_r, est, _ in predictions:
        if not uid in top_n:
            top_n[uid] = []
        top_n[uid].append((iid, est))

    # Sort the predictions for each user and retrieve the n highest ones
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

# Get top 10 recommendations for each user
top_n_recommendations = get_top_n_recommendations(predictions, n=10)

# Display recommendations for a specific user
user_id = 1  # Replace with an actual user ID
print(f"Top 10 recommendations for user {user_id}: {top_n_recommendations.get(user_id, [])}")


