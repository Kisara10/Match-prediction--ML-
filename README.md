# Match-prediction--ML-
# Cricket Match Winner Prediction (Machine Learning)

This project predicts the winner of ODI cricket matches using Machine Learning techniques.  
The model is trained on historical ODI match data from ESPN Cricinfo.

## Dataset
The dataset contains ODI match results including:

- Team 1
- Team 2
- Ground
- Winner

Matches range from 1971 to 2019.

## Data Preprocessing
The following preprocessing steps were applied:

1. Removed irrelevant columns (Margin, Scorecard)
2. Selected key features (Team 1, Team 2, Ground)
3. Converted categorical variables into numerical values using LabelEncoder
4. Split the dataset into training (80%) and testing (20%)

## Machine Learning Models Used

Three models were trained and compared:

- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)

## Model Performance

| Model | Accuracy |
|------|------|
| Logistic Regression | 21.6% |
| Decision Tree | 54.9% |
| KNN | 42.3% |

Decision Tree achieved the highest accuracy and was selected as the final model.

## Final Model

The final model used for prediction is:

Decision Tree Classifier

The trained model is saved as:

cricket_match_predictor.pkl

## Prediction Function

A function was implemented to predict match outcomes:

predict_match(team1, team2, ground)

Example:

predict_match("India", "Australia", "Melbourne")

## Technologies Used

- Python
- Pandas
- Scikit-learn

## Future Improvements

- Add more match features (toss winner, team ranking, recent performance)
- Improve prediction accuracy with feature engineering
- Build a web interface for real-time predictions

