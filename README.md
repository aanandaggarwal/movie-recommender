# Movie Recommender System Using SVD

This repository contains a movie recommender system I built using collaborative filtering, implemented in Python. The model utilizes the MovieLens dataset to provide personalized movie recommendations based on user ratings, with an additional explainability feature to make recommendations more transparent.

## Overview
The model is trained on the MovieLens dataset, which contains user ratings, movie information, and user-generated tags. The recommender system aims to predict user preferences and generate top movie recommendations. 

## Features
- **Collaborative Filtering with SVD**: Predicts user preferences for unrated movies by learning latent features from the user-item interaction matrix.
- **K-Fold Cross-Validation**: Evaluates model performance using cross-validation to ensure robust results.
- **Explainability**: Provides explanations for recommended movies based on genres and user-applied tags to enhance transparency.

## Dataset
This project utilizes the MovieLens dataset, which includes several files:
- `ratings.csv`: User ratings for movies.
- `movies.csv`: Movie metadata such as titles and genres.
- `tags.csv`: User-generated tags for movies.
- `links.csv`: Movie links to external databases (not used in the current version).

Ensure these files are available in the `data/` folder within the project directory.

## Setup and Installation
### Prerequisites
- Python 3.x
- Required libraries listed in `requirements.txt`

### Installation
1. **Clone the Repository**:
   ```sh
   git clone https://github.com/yourusername/movie-recommender.git
   cd movie-recommender
   ```

2. **Set Up Environment**:
   Create a virtual environment and install the required dependencies:
   ```sh
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. **Organize the Data**:
   Place the dataset files (`ratings.csv`, `movies.csv`, `tags.csv`, `links.csv`) in a folder named `data/` at the root level of the repository.

## Running the Model
To run the movie recommender system:
1. Ensure that you have activated your virtual environment:
   ```sh
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

2. Run the script:
   ```sh
   python movie_recommender.py
   ```

### Script Overview
- **Data Preparation**: Loads the datasets and splits the ratings into training and testing sets.
- **Training with SVD**: Uses the SVD algorithm from the Surprise library to learn latent factors from user-item ratings.
- **Cross-Validation**: Performs 5-fold cross-validation to evaluate model performance in terms of Mean Absolute Error (MAE) and Root Mean Square Error (RMSE).
- **Top-N Recommendations**: Generates top 10 movie recommendations for each user and evaluates them using metrics like Precision, Recall, F-measure, and NDCG.
- **Explainability**: Provides user-specific explanations for recommendations based on genres and user-applied tags.

## Evaluation
The script prints evaluation metrics to help assess the performance of the model:
- **MAE** and **RMSE**: Measure the accuracy of predicted ratings.
- **Precision**, **Recall**, **F-measure**, **NDCG**: Assess the quality of the generated recommendations.

## Acknowledgements
Special thanks to Professor Zhang and the course staff for their guidance and support. The MovieLens dataset used in this project was introduced by F. Maxwell Harper and Joseph A. Konstan (2015) in "The MovieLens Datasets: History and Context." Additional thanks to all the helpful teachers online for their tutorials and resources that supported the technical implementation of this project.