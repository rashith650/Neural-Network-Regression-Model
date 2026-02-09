# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

This code builds and trains a feedforward neural network in PyTorch for a regression task. The model takes a single input feature, passes it through two hidden layers with ReLU activation, and predicts one continuous output. It uses MSE loss and RMSProp optimizer to minimize the error between predictions and actual values over training epochs.

## Neural Network Model

<img width="1059" height="642" alt="Screenshot 2026-02-09 114708" src="https://github.com/user-attachments/assets/6bf0e5bf-5c18-49d9-a852-2d7c225dcf20" />

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
### Register Number:212223243003
```python
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
       elf.fc1 = nn.Linear(1, 8)
        self.fc2 = nn.Linear(8, 10)
        self.fc3 = nn.Linear(10, 1)

        self.relu = nn.ReLU()
        self.history = {'loss': []}

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the Model, Loss Function, and Optimizer
ai_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(ai_brain.parameters(), lr=0.001)



def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
   for epoch in range(epochs):
        optimizer.zero_grad()

        output = model(X_train)
        loss = criterion(output, y_train)

        loss.backward()
        optimizer.step()

        model.history['loss'].append(loss.item())

        if epoch % 200 == 0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}")



train_model(ai_brain, X_train_tensor, y_train_tensor, criterion, optimizer)



with torch.no_grad():
    test_loss = criterion(ai_brain(X_test_tensor), y_test_tensor)
    print(f"Test Loss: {test_loss.item():.6f}")



loss_df = pd.DataFrame(ai_brain.history)

loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()


X_new = torch.tensor([[9]], dtype=torch.float32)

X_new_scaled = scaler.transform(X_new)
X_new_tensor = torch.tensor(X_new_scaled, dtype=torch.float32)

prediction = ai_brain(X_new_tensor).item()

print(f"Prediction: {prediction}")



```
## Dataset Information

<img width="675" height="627" alt="Screenshot 2026-02-09 103525" src="https://github.com/user-attachments/assets/bc44959e-e94f-48b2-a10c-ef3727f28690" />

## OUTPUT
<img width="581" height="257" alt="Screenshot 2026-02-09 111753" src="https://github.com/user-attachments/assets/24612f31-69a8-469b-959f-0b9b93fc7a3b" />


### Training Loss Vs Iteration Plot
<img width="863" height="613" alt="Screenshot 2026-02-09 111839" src="https://github.com/user-attachments/assets/5ed57b15-4c21-493b-b0f0-b7e88ba44ed8" />



### New Sample Data Prediction

<img width="441" height="44" alt="image" src="https://github.com/user-attachments/assets/8aa1970b-72eb-4f0b-92a8-e8497fc8647d" />


## RESULT

Neural network regression model for the given dataset has been done successfully.
