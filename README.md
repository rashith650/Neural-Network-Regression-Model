# Developing a Neural Network Regression Model
## NAME: MOHAMED RASHITH S
## REG NO:212223243003
## AIM

To develop a neural network regression model for the given dataset.

## THEORY

This code builds and trains a feedforward neural network in PyTorch for a regression task.
The model takes a single input feature, passes it through two hidden layers with ReLU activation, and predicts one continuous output.
It uses MSE loss and RMSProp optimizer to minimize the error between predictions and actual values over training epochs.

## Neural Network Model

<img width="930" height="643" alt="image" src="https://github.com/user-attachments/assets/eaebc717-17d1-4762-89a3-b584d2b05979" />

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:MOHAMED RASHITH S
### Register Number: 212223243003
```
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def rash():
    print("Name: MOHAMED RASHITH S")
    print("Register Number: 212223243003")

dataset1 = pd.read_csv('/content/MyMLData.csv')

X = dataset1[['Input']].values
y = dataset1[['Output']].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=33
)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1,1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1,1)

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 6)
        self.fc3 = nn.Linear(6, 1)
        self.relu = nn.ReLU()
        self.history = {'loss': []}
        rash()
        print("Neural Network Regression Model Initialized")

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

ai_rash = NeuralNet()

criterion = nn.MSELoss()
optimizer = optim.Adam(ai_rash.parameters(), lr=0.01)

def train_model(ai_rash, X_train, y_train, criterion, optimizer, epochs=2000):
    rash()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = ai_rash(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        ai_rash.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

train_model(ai_rash, X_train_tensor, y_train_tensor, criterion, optimizer)

with torch.no_grad():
    rash()
    test_loss = criterion(ai_rash(X_test_tensor), y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')

loss_df = pd.DataFrame(ai_rash.history)

loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()

X_n1_1 = torch.tensor([[9]], dtype=torch.float32)
prediction = ai_rash(torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)).item()

rash()
print(f'Prediction for input 9: {prediction}')


```
## Dataset Information

<img width="410" height="503" alt="image" src="https://github.com/user-attachments/assets/c8b4f75b-d1a3-4351-965f-3bfd8bebbc50" />


## OUTPUT

<img width="474" height="290" alt="image" src="https://github.com/user-attachments/assets/7582fe28-9e4e-4d09-9765-e13614d63b14" />


### Training Loss Vs Iteration Plot
<img width="796" height="426" alt="image" src="https://github.com/user-attachments/assets/834b0772-efb4-4ca7-ab23-7f0cfd3e8c30" />



### New Sample Data Prediction

<img width="300" height="62" alt="image" src="https://github.com/user-attachments/assets/d092c69e-b7e6-4db3-a826-849f30113535" />


## RESULT

Successfully executed the code to develop a neural network regression model.

