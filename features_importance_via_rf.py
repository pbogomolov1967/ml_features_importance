# Feature importance via Random Forest Ensemble Classifier

import pandas as pd
from util_cols_importance import calc_importance

df_name = "data/credit_history.csv"
df = pd.read_csv(df_name)
col_names1 = list(df.columns)

data = df.dropna()
data = pd.get_dummies(data)

X = data.copy()
y = X['default']
X.drop('default', axis=1, inplace=True)
col_names2 = list(X.columns)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=123,
                                                    train_size=0.7,
                                                    test_size=0.3)
# Train the RandomForestRegressor on our data
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(random_state=123, n_estimators=10)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(f'Model Score: {model.score(X_test, y_test)}, Sum of Importances: {sum(model.feature_importances_)}')

features, importance_percent = calc_importance(col_names2, model.feature_importances_, show_prc=50, collapse_vals=True)
print(list(zip(features, importance_percent)))

print('done')
