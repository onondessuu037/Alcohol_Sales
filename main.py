# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# This relates to plotting datetime values with matplotlib:
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

df = pd.read_csv('Alcohol_Sales.csv',index_col=0,parse_dates=True)
print(len(df))

# Always a good idea with time series data:
df.dropna(inplace=True)
print(len(df))

print(df.head())
print(df.tail())


plt.figure(figsize=(12,4))
plt.title('Beer, Wine, and Alcohol Sales')
plt.ylabel('Sales (millions of dollars)')
plt.grid(True)
plt.autoscale(axis='x',tight=True)
plt.plot(df['S4248SM144NCEN'])
plt.show()

# Extract values from the source .csv file
y = df['S4248SM144NCEN'].values.astype(float)

# Define a test size
test_size = 12

# Create train and test sets
train_set = y[:-test_size]
test_set = y[-test_size:]
print(test_set)
"""
test_set
[10415. 12683. 11919. 14138. 14583. 12640. 14257. 12396. 13914. 14174...
 
"""
print(test_set.reshape(-1, 1))
""" 
test_set.reshape(-1, 1)
[[10415.]
 [12683.]
 [11919.]
 [14138.]
 [14583.]
 [12640.]
 ...
"""

from sklearn.preprocessing import MinMaxScaler

# Instantiate a scaler with a feature range from -1 to 1
# 前處理
scaler = MinMaxScaler(feature_range=(-1, 1))
print(scaler)
# Normalize the training set
train_norm = scaler.fit_transform(train_set.reshape(-1, 1))  #reshape 格式
# print(train_norm) # train_norm is after normalization
print(train_norm.min())
print(train_norm.max())
print(train_norm.mean())

# print(train_norm)
#Prepare data for LSTM
# Convert train_norm from an array to a tensor
train_norm = torch.FloatTensor(train_norm).view(-1)

# Define a window size
window_size = 12

# Define function to create seq/label tuples
def input_data(seq,ws):  # ws is the window size
    out = []
    L = len(seq)
    for i in range(L-ws):
        window = seq[i:i+ws]
        label = seq[i+ws:i+ws+1]
        out.append((window,label))
    return out

# Apply the input_data function to train_norm
train_data = input_data(train_norm,window_size)
print(len(train_data))  # this should equal 325-12-12
# print(train_data)  # this should equal 325-12-12

# Display the first seq/label tuple in the train data
print(train_data[0].__getitem__(0).view(12, -1))
"""
print(train_data[0].__getitem__(0).view(12, 1, -1))
tensor([[[-0.9268]],

        [[-0.9270]],

        [[-0.8340]],

        [[-0.7379]],
"""
print("====================")

class LSTMnetwork(nn.Module):
    def __init__(self, input_size=1, hidden_size=200, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size

        # Add an LSTM layer:
        self.lstm = nn.LSTM(input_size, hidden_size)

        # Add a fully-connected layer:
        self.linear = nn.Linear(hidden_size, output_size)

        # Initialize h0 and c0:
        self.hidden = (torch.zeros(1, 1, self.hidden_size),
                       torch.zeros(1, 1, self.hidden_size))

    def forward(self, seq):
        ## seq.view reshape size
        lstm_out, self.hidden = self.lstm(
            seq.view(len(seq), 1, -1), self.hidden)  # view:reshape
        pred = self.linear(lstm_out.view(len(seq), -1))
        return pred[-1]  # we only want the last value

torch.manual_seed(101)
model = LSTMnetwork()
print(f"**** {model.hidden}")
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print(model)


def count_parameters(model):
    params = [p.numel() for p in model.parameters() if p.requires_grad]
    for item in params:
        print(f'{item:>6}')
    print(f'______\n{sum(params):>6}')
"""
LSTMnetwork(
  (lstm): LSTM(1, 200)
  (linear): Linear(in_features=200, out_features=1, bias=True)
)
   800
160000
   800
   800
   200
     1
______
162601
"""

print(count_parameters(model))
## Train the model
epochs = 50

import time

start_time = time.time()

for epoch in range(epochs):

    # extract the sequence & label from the training data
    for seq, y_train in train_data:
        # reset the parameters and hidden states
        optimizer.zero_grad()
        model.hidden = (torch.zeros(1, 1, model.hidden_size),
                        torch.zeros(1, 1, model.hidden_size))

        y_pred = model(seq)

        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

    # print training result
    print(f'Epoch: {epoch + 1:2} Loss: {loss.item():10.8f}')

print(f'\nDuration: {time.time() - start_time:.0f} seconds')
## Run predictions and compare to known test set

future = 12

# Add the last window of training values to the list of predictions
preds = train_norm[-window_size:].tolist()

# Set the model to evaluation mode
model.eval()
print(f"model.hidden_size {model.hidden_size}")
for i in range(future):
    seq = torch.FloatTensor(preds[-window_size:])
    with torch.no_grad():
        model.hidden = (torch.zeros(1,1,model.hidden_size),
                        torch.zeros(1,1,model.hidden_size))
        preds.append(model(seq).item())

# Display predicted values
print(preds[window_size:]) # equivalent to preds[-future:]

true_predictions = scaler.inverse_transform(np.array(preds[window_size:]).reshape(-1, 1))
print(true_predictions)

print(df['S4248SM144NCEN'][-12:])


# Remember that the stop date has to be later than the last predicted value.
x = np.arange('2018-02-01', '2019-02-01', dtype='datetime64[M]').astype('datetime64[D]')
print(x)

plt.figure(figsize=(12,4))
plt.title('Beer, Wine, and Alcohol Sales')
plt.ylabel('Sales (millions of dollars)')
plt.grid(True)
plt.autoscale(axis='x',tight=True)
plt.plot(df['S4248SM144NCEN'])
plt.plot(x,true_predictions)
plt.show()


# Plot the end of the graph
fig = plt.figure(figsize=(12,4))
plt.title('Beer, Wine, and Alcohol Sales')
plt.ylabel('Sales (millions of dollars)')
plt.grid(True)
plt.autoscale(axis='x',tight=True)
fig.autofmt_xdate()

# Select the end of the graph with slice notation:
plt.plot(df['S4248SM144NCEN']['2017-01-01':])
plt.plot(x,true_predictions)
plt.show()


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
