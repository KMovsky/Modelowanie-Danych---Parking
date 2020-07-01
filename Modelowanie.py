import numpy as np #listy i arraye
import os #obsluga systemu i plikow
import matplotlib.pyplot as plt
import sklearn #sci-kit learn
import matplotlib.pyplot as plt
import random
import re #regular expressions

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

#dataset.csv:
#SystemCodeNumber,Capacity,Occupancy,LastUpdated

filepath = 'dataset-parking.csv'

datafile = open(filepath, 'r')
lines=datafile.readlines()


y = []
Parking_Spots = []

for line in lines:

    if line.split(',')[0] not in Parking_Spots:
        Parking_Spots.append(line.split(',')[0])

    if line.split(',')[0] == Parking_Spots[0]:
        lineVar = line.split(',')[2]
        y.append(int(lineVar))

    

# Create the dataset
rng = np.random.RandomState(1)
x = np.linspace(0, 1312, 1312)[:, np.newaxis]
#y = np.sin(X).ravel() + np.sin(6 * X).ravel() + rng.normal(0, 0.1, X.shape[0])

print(y)

# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=4)

regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                          n_estimators=300, random_state=rng)

regr_1.fit(x, y)
regr_2.fit(x, y)

# Predict
y_1 = regr_1.predict(x)
y_2 = regr_2.predict(x)

# Plot the results
plt.figure()
plt.scatter(x, y, c="k", label="training samples")
plt.plot(x, y_1, c="g", label="n_estimators=1", linewidth=2)
plt.plot(x, y_2, c="r", label="n_estimators=300", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Boosted Decision Tree Regression")
plt.legend()
plt.show()


datafile.close()