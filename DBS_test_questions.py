""" 
Answers to DBS's technical test for role of Data Scientist
by: Lin Junyuan
"""
# import libraries
import numpy as np
import pandas as pd
from scipy.integrate import dblquad

# Qn 1 - Find P (X < Y)

prob = dblquad(lambda x, y: np.exp(-x-y), 0, np.inf, lambda x: 0, lambda x: x)

print("\nQn 1 - Find P(X < Y)")
print(f"P(X < Y): {round(prob[0], 2)}\n")

# Qn 2 - Counting pairs
print("Qn 2 - Find pairs")

## Get user input
input_list = list(map(int,input("Enter numbers : ").strip().split()))
input_list.sort()
diff = int(input("Enter difference: "))

ans = set()

for i in range(len(input_list)-1):
    if input_list[i+1] - input_list[i] == diff:
        ans.add((input_list[i], input_list[i+1]))

print(f"Detected pairs: {ans}\n")

# Qn 3 - Return list of indices

data = [
 ("username1","phone_number1", "email1"),
 ("usernameX","phone_number1", "emailX"),
 ("usernameZ","phone_numberZ", "email1Z"),
 ("usernameY","phone_numberY", "emailX"),
 ]

indice_list = []

for i in range(len(data)):
    # Check if list is empty
    if len(indice_list) == 0:
        # Add first element in data as a group in indice list
        indice_list.append([i])
    else:
        assigned = False
        # Check if element in data is same person as anyone already in indice list
        for group in indice_list:
            for j in group:
                # If found to be same person, add to group
                if data[i][0] == data[j][0] or data[i][1] == data[j][1] or data[i][2] == data[j][2]:
                    group.append(i)
                    assigned = True
                    break
            # break loop if element is assigned in previous iteration                       
            if assigned:
                break
        # add new group if found to be a new person                   
        if not assigned:
                indice_list.append([i])               

print(indice_list)

# Qn 4 - Implement Forward propagation and Backward Propagation in a 3 layered Neural Network

## Import Libraries for Neural Network
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Generate random regression problem
X, y = make_regression(n_samples=10_000, n_features=20, random_state=42)

# Perform train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Perform Standard scaling
ss = StandardScaler()
X_train_sc = ss.fit_transform(X_train)
X_test_sc = ss.transform(X_test)

# Initialize model
model = Sequential()
# Add three layers with random weights and bias initialized
model.add(Dense(15, input_shape=(20,), kernel_initializer='random_uniform', bias_initializer='random_normal' ,activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(15, activation='relu'))
# Add output layer
model.add(Dense(1))

# Compile model
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

# Fit model
model.fit(X_train_sc, y_train, epochs=40, batch_size=256, validation_data=(X_test_sc, y_test))
