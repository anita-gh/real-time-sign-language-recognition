import os
import pickle 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# finding dynamic path of the current Python file
script_dir = os.path.dirname(os.path.realpath(__file__))
pickle_path = os.path.join(script_dir, 'l_hands_points.pickle')

print(f"Looking for pickle file at:\n{pickle_path}\n") #debug

# load binary file
if not os.path.exists(pickle_path):
    print("error:l_hands_points.pickle not found! Please run creating_datasets.py first.")
    exit()

datadict = pickle.load(open(pickle_path, 'rb'))

# convert data to NumPy array
data = np.asarray(datadict['data'])
barchasb = np.asarray(datadict['label'])

# to ensure 84 column data is loaded
unique, counts = np.unique(barchasb, return_counts=True)
print("data status")
print(f"total images: {len(data)}")
print(f"data shape (samples x features): {data.shape}") 
for l, c in zip(unique, counts):
    print(f"class {l}: {c} samples")
print("-" * 50)

# data segmentation
if min(counts) < 2:
    print("warning disabling 'stratify' due to low data counts.")
    x_train, x_test, y_train, y_test = train_test_split(
        data, barchasb, test_size=0.2, shuffle=True
    )
else:
    x_train, x_test, y_train, y_test = train_test_split(
        data, barchasb, test_size=0.2, shuffle=True, stratify=barchasb
    )

# building and training a random forest model
fmodel = RandomForestClassifier()
print("training the 2-hand random forest model...")
fmodel.fit(x_train, y_train)

# new accuracy prediction and calculation
y_predict = fmodel.predict(x_test)
score = accuracy_score(y_predict, y_test)

print(f"Accuracy: {score * 100:.2f}% of samples were classified correctly!")

# save the final model in the script folder
model_save_path = os.path.join(script_dir, 'fmodel.pickle')
with open(model_save_path, 'wb') as f:
    pickle.dump({'fmodel': fmodel}, f)

print(f"Model successfully saved at:\n{model_save_path}")