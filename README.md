# Movie Recommender System Using SVD

This repository contains a collaborative filtering-based movie recommender system built in Python. The model utilizes the MovieLens dataset to provide personalized movie recommendations and includes an explainability feature to make recommendations more transparent to users.

## Highlights
- **Collaborative Filtering with SVD**: Predicts user preferences for unrated movies by learning latent features from the user-item interaction matrix.
- **K-Fold Cross-Validation**: Evaluated model performance using cross-validation to ensure robust results.
- **Explainability**: Provides explanations for recommended movies based on genres and user-applied tags to enhance transparency.

## Dataset

This project leverages the MovieLens dataset, which includes several key files. If the datasets are not already present in the `data/` folder after cloning, you can download them from the MovieLens website:

- [ratings.csv](https://grouplens.org/datasets/movielens/latest/): User ratings for movies.
- [movies.csv](https://grouplens.org/datasets/movielens/latest/): Movie metadata such as titles and genres.
- [tags.csv](https://grouplens.org/datasets/movielens/latest/): User-generated tags for movies.
- [links.csv](https://grouplens.org/datasets/movielens/latest/): Movie links to external databases (not used in the current version).

Place these files in a folder named `data/` at the root of the project directory.

## Setup and Installation
### Prerequisites
- Python 3.x
- Required libraries listed in `requirements.txt`

### Installation
1. **Clone the Repository**:
   ```sh
   git clone https://github.com/aanandaggarwal/movie-recommender.git
   cd movie-recommender
   ```

2. **Set Up Environment**:
   Create a virtual environment and install the required dependencies:
   ```sh
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. **Prepare the Data**:
   If the dataset files are not included in the cloned repository, download the following files ([ratings.csv, movies.csv, tags.csv, links.csv](https://grouplens.org/datasets/movielens/latest/)) from the MovieLens dataset and place them in a data/ folder at the root of the project.

## Key Dependencies

This project uses several Python packages:

- **Pandas**: Used for data manipulation and analysis, including loading and processing the MovieLens dataset.
- **NumPy**: Provides support for numerical operations, such as calculating metrics efficiently.
- **scikit-learn**: Used for splitting the dataset into training/testing sets and providing utilities for model evaluation.
- **Surprise**: A library specialized for building recommender systems. It is used to implement the SVD algorithm, perform cross-validation, and evaluate the model.
- **Collections (defaultdict)**: Used to efficiently store and manipulate lists of items such as recommendations for users.

These dependencies are included in the `requirements.txt` file, which ensures that the necessary versions are installed for running the project smoothly.

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
   
## Evaluation

The recommender system is evaluated using several key metrics:
- **Mean Absolute Error (MAE)**: Measures the average absolute difference between actual and predicted ratings, indicating prediction accuracy. **Lower MAE values** indicate better accuracy of the model in predicting ratings close to the actual values.
- **Root Mean Square Error (RMSE)**: Similar to MAE but gives more weight to larger errors by squaring the differences between actual and predicted ratings. RMSE penalizes significant deviations, providing a more sensitive measure of accuracy. **Lower RMSE values** indicate fewer large errors in the model's predictions.
- **Precision**: Measures the proportion of recommended movies that are relevant. **Higher precision** means fewer irrelevant recommendations.
- **Recall**: Measures the proportion of relevant movies that were successfully recommended. **Higher recall** means more comprehensive recommendations.
- **F-measure**: Balances precision and recall, providing a single value that helps evaluate the model's ability to recommend relevant items.
- **Normalized Discounted Cumulative Gain (NDCG)**: Evaluates how well relevant items are ranked in the recommendation list, with **higher values** indicating better-ranked recommendations.

## Acknowledgements
Special thanks to Professor Zhang and the course staff for their guidance and support. The MovieLens dataset used in this project was introduced by F. Maxwell Harper and Joseph A. Konstan (2015) in "The MovieLens Datasets: History and Context." Additional thanks to all the helpful instructors/creators online for their tutorials and resources that supported the technical implementation of this project.
