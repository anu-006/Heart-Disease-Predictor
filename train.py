import pickle
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    PowerTransformer,
)
from sklearn.linear_model import LogisticRegression

df = pd.read_pickle('data.pkl')

if 'target' in df.columns:
    target_col = 'target'
elif 'disease' in df.columns:
    target_col = 'disease'
else:
    raise ValueError('no target column found in data')

X = df.drop(target_col, axis=1)
y = df[target_col]

# feature lists – these may be larger than what actually exists
categorical_features = [
    'sex', 'cp', 'fbs', 'restecg', 'exang',
    'slope', 'ca', 'thal'
]
numerical_features_power_and_scale = ['trestbps', 'chol', 'oldpeak']
numerical_features_scale_only = ['age', 'thalach']

# keep only the columns that are present in X
categorical_features = [c for c in categorical_features if c in X.columns]
numerical_features_power_and_scale = [
    c for c in numerical_features_power_and_scale if c in X.columns
]
numerical_features_scale_only = [
    c for c in numerical_features_scale_only if c in X.columns
]

preprocessor = ColumnTransformer(
    transformers=[
        ('num_power_scale', Pipeline([
            ('power', PowerTransformer(method='yeo-johnson')),
            ('scaler', StandardScaler()),
        ]), numerical_features_power_and_scale),
        ('num_scale', StandardScaler(), numerical_features_scale_only),
        ('cat_ohe', OneHotEncoder(
            handle_unknown='ignore',
            # use sparse=False if sklearn<1.2
            sparse_output=False,
            drop='first'
        ), categorical_features),
    ],
    remainder='passthrough',
)
# remove the next line if your sklearn version is <1.2
try:
    preprocessor = preprocessor.set_output(transform='pandas')
except AttributeError:
    pass

pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LogisticRegression(solver='liblinear')),
])

pipe.fit(X, y)

with open('pipe.pkl', 'wb') as f:
    pickle.dump(pipe, f)

print('training complete, pipeline saved to pipe.pkl')
