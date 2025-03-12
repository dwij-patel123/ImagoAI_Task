from catboost import CatBoostRegressor
from preprocess import process_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score,root_mean_squared_error
import matplotlib.pyplot as plt
import pickle
import datetime

path = 'data/TASK-ML-INTERN.csv'
(X,y),(X_scaled,y) = process_data(path)

# As we know from the notebook experimentation the best model is Catboost
# with parameters {'depth': 3, 'iterations': 7, 'learning_rate': 1} so we use that to train

def train():
    # split data into train(80%) and test(20%)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model_cat = CatBoostRegressor(depth=3,
                                  iterations=7,
                                  learning_rate=1,
                                  loss_function='RMSE')
    model_cat.fit(X_train,y_train)
    preds = model_cat.predict(X_test)
    mae_loss = mean_absolute_error(y_test,preds)
    rmse_loss = root_mean_squared_error(y_test,preds)
    r2_cat = r2_score(y_test,preds)
    print(f'Mean Absolute Error of Catboost Model:{mae_loss}')
    print(f'Root Mean Square Error of Catboost model:{rmse_loss}')
    print(f'R2 Score of Catboost model:{r2_cat}')

    # visualize the scatter plot between predictions vs y_true
    plt.figure(figsize=(8, 6))
    plt.scatter(preds, y_test, alpha=0.6,label='CatBoost Predictions')
    plt.xlabel("Predictions")
    plt.ylabel("True Labels")
    plt.title("Predictions VS True Labels")
    plt.legend()
    plt.show()


    # Saving model using Pickle
    with open(f'models/model_{datetime.datetime.now()}.pkl', 'wb') as f:
        pickle.dump(model_cat, f)

train()



