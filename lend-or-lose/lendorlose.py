# # Import necessary libraries
# import pandas as pd
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import accuracy_score
# from sklearn.tree import DecisionTreeClassifier
# from xgboost import XGBClassifier
# from sklearn.ensemble import GradientBoostingClassifier

# # Load the dataset
# data = pd.read_csv("train.csv")

# # Label encoding for categorical features
# categorical_features = ['Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner']
# label_encoders = {}

# for col in categorical_features:
#     le = LabelEncoder()
#     data[col] = le.fit_transform(data[col])
#     label_encoders[col] = le  # Store the encoder for possible inverse transformation

# # Split data into features (X) and target (y)
# X = data.drop(columns=['LoanID', 'Default'])  # Drop ID column and target
# y = data['Default']

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Function to perform hyperparameter tuning and model training
# def train_and_tune_model(model, param_grid):
#     grid_search = GridSearchCV(model, param_grid, scoring='accuracy', cv=5, n_jobs=-1, verbose=2)
#     grid_search.fit(X_train, y_train)
#     best_model = grid_search.best_estimator_
#     return best_model

# # Define hyperparameter grids for each model
# dt_params = {
#     'max_depth': [5, 10, 15, 20, None],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 5]
# }

# xgb_params = {
#     'learning_rate': [0.01, 0.1, 0.2],
#     'max_depth': [3, 6, 10],
#     'n_estimators': [50, 100, 200]
# }

# gb_params = {
#     'learning_rate': [0.01, 0.1, 0.2],
#     'n_estimators': [50, 100, 200],
#     'max_depth': [3, 5, 7]
# }

# # Train and tune Decision Tree model
# dt_model = train_and_tune_model(DecisionTreeClassifier(random_state=42), dt_params)
# print(f"Best Decision Tree Model: {dt_model}")

# # Train and tune XGBoost model
# xgb_model = train_and_tune_model(XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42), xgb_params)
# print(f"Best XGBoost Model: {xgb_model}")

# # Train and tune Gradient Boosting model
# gb_model = train_and_tune_model(GradientBoostingClassifier(random_state=42), gb_params)
# print(f"Best Gradient Boosting Model: {gb_model}")

# # Evaluate models
# def evaluate_model(model, X_test, y_test):
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f"Accuracy: {accuracy:.4f}")

# print("\nEvaluating Decision Tree Model:")
# evaluate_model(dt_model, X_test, y_test)

# print("\nEvaluating XGBoost Model:")
# evaluate_model(xgb_model, X_test, y_test)

# print("\nEvaluating Gradient Boosting Model:")
# evaluate_model(gb_model, X_test, y_test)



# Import necessary libraries
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Load the dataset
data = pd.read_csv("train.csv")

# Label encoding for categorical features
categorical_features = ['Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner']
label_encoders = {}

for col in categorical_features:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le  # Store the encoder for possible inverse transformation

# Split data into features (X) and target (y)
X = data.drop(columns=['LoanID', 'Default'])  # Drop ID column and target
y = data['Default']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a callback function to capture and log accuracies only
def log_best_score(grid_search):
    best_score = grid_search.best_score_
    print(f"Best accuracy for model: {best_score:.4f}")

# Define hyperparameter grids for each model
dt_params = {
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5]
}

xgb_params = {
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 6, 10],
    'n_estimators': [50, 100, 200]
}

gb_params = {
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7]
}

# Train and tune Decision Tree model
dt_grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), dt_params, scoring='accuracy', cv=5, n_jobs=-1, verbose=0)
dt_grid_search.fit(X_train, y_train)
log_best_score(dt_grid_search)

# Train and tune XGBoost model
xgb_grid_search = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42), xgb_params, scoring='accuracy', cv=5, n_jobs=-1, verbose=0)
xgb_grid_search.fit(X_train, y_train)
log_best_score(xgb_grid_search)

# Train and tune Gradient Boosting model
gb_grid_search = GridSearchCV(GradientBoostingClassifier(random_state=42), gb_params, scoring='accuracy', cv=5, n_jobs=-1, verbose=0)
gb_grid_search.fit(X_train, y_train)
log_best_score(gb_grid_search)
