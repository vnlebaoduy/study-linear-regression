from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error

def find_best_alpha(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Lasso()
    parameters = {'alpha': [0.001,0.002,0.003, 0.01,0.02,0.03, 0.1,0.2,0.3, 1,2,3, 10,20,30, 100]}
    grid_search = GridSearchCV(estimator=model, param_grid=parameters, scoring='neg_mean_squared_error', cv=5)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    print("Best alpha:", grid_search.best_params_['alpha'])
    print("Mean Squared Error on Test set:", mse)
    
    return best_model


from sklearn.datasets import load_diabetes
data = load_diabetes()
X, y = data.data, data.target
best_lasso_model = find_best_alpha(X, y)
