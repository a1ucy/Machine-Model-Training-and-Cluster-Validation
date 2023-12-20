import pickle
import pandas as pd
from train import extract_features, normalize, get_PCA

model = pickle.load(open( "model.pkl", "rb" ))

data = pd.read_csv('test.csv', header = None, low_memory = False).values.tolist()
df = extract_features(data)
df = normalize(df)
pca = get_PCA()
df = pca.transform(df)

X_test = pd.DataFrame(df)
y_predictions = model.predict(X_test)
y_df = pd.DataFrame(y_predictions)
y_df.to_csv('Result.csv', index = False, header = False)