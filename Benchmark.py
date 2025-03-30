from typing import Sequence
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import random


# load training and test data
def loadData():
    X_train = np.load('X_train.npy',allow_pickle=True)
    y_train = np.load('y_train.npy',allow_pickle=True)
    X_test = np.load('X_test.npy',allow_pickle=True)
    y_test = np.load('y_test.npy',allow_pickle=True)

    X_train = [torch.Tensor(x) for x in X_train]  # List of Tensors (SEQ_LEN[i],INPUT_DIM) i=0..NUM_SAMPLES-1
    X_test = [torch.Tensor(x) for x in X_test]  # List of Tensors (SEQ_LEN[i],INPUT_DIM)
    y_train = torch.Tensor(y_train) # (NUM_SAMPLES,1)
    y_test = torch.Tensor(y_test) # (NUM_SAMPLES,1)

    return X_train, X_test, y_train, y_test



# Define a Vanilla RNN layer by hand
class RNNLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNNLayer, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.W_xh = nn.Parameter(torch.randn(hidden_size, input_size) * 0.01)
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)

        #For Vanila RNNs the activation function is typically tanh
        self.activation = torch.tanh

    def forward(self, x, hidden):
        # h(t) = tanh(W_xh * x(t) + W_hh * h(t-1))
        hidden = self.activation(
            torch.matmul(self.W_xh, x) +
            torch.matmul(self.W_hh, hidden)
        )

        return hidden

# Define a sequence prediction model using the Vanilla RNN
class SequenceModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SequenceModel, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = RNNLayer(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, seq_lengths):
        batch_size = len(input_seq)

        #Tensor to store last hidden states
        last_hidden = torch.zeros(batch_size, self.hidden_size).to(input_seq[0].device)

        for b in range(batch_size):
            # Initialize hidden state for this sequence
            hidden = torch.zeros(self.hidden_size).to(input_seq[0].device)

            seq_length = input_seq[b].shape[0]

            #Process each time step in the sequence
            for t in range(seq_length):
                hidden = self.rnn(input_seq[b][t], hidden)
            
            # Store the last hidden state in the output tensor
            last_hidden[b] = hidden

        output = self.linear(last_hidden)
        return output

# Define a sequence prediction model for fixed length sequences, BUT NO SHARED WEIGHTS
class SequenceModelFixedLen(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, seq_len):
        super(SequenceModelFixedLen, self).__init__()
        self.hidden_size = hidden_size
        self.seq_len = seq_len

        #Use separate RNN layers for each time step
        self.rnn_layers = [RNNLayer(input_size, hidden_size) for _ in range(seq_len)]
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, seq_lengths):
        batch_size = len(input_seq)
        last_hidden = torch.zeros(batch_size, self.hidden_size).to(input_seq[0].device)

        for b in range(batch_size):
            #Initialize the hidden state
            hidden = torch.zeros(self.hidden_size).to(input_seq[0].device)

            #Use minimum of sequence length and predefined length
            seq_length = min(input_seq[b].shape[0], self.seq_len)
            for t in range(seq_length):
                #Use corresponding RNN layer for this time step
                hidden = self.rnn_layers[t](input_seq[b][t], hidden)
            
            # Store the last hidden state in the output tensor
            last_hidden[b] = hidden

        output = self.linear(last_hidden)
        return output

def pad_inputs(input_sequences, max_length):
    #Store original lengths
    seq_lengths = torch.tensor([seq.shape[0] for seq in input_sequences])

    #Pad sequences to max_length
    padded_sequences = pad_sequence(input_sequences, batch_first=True)

    #Ensure all sequences are of max_length
    if padded_sequences.shape[1] < max_length:
        padding = torch.zeros((padded_sequences.shape[0], max_length - padded_sequences.shape[1], padded_sequences.shape[2]))
        padded_sequences = torch.cat((padded_sequences, padding), dim=1)

    return padded_sequences, seq_lengths



# load data
X_train, X_test, y_train, y_test = loadData()
device = y_train.device

# Define hyperparameters and other settings
input_size = X_train[0].shape[1]  # Replace with the actual dimension of your input features
hidden_size = 64
output_size = 1
num_epochs = 20
learning_rate = 0.001
batch_size = 32

# Create the model using min length input
seq_lengths = [seq.shape[0] for seq in X_train]

#Find minimum and maximum lengths
min_length = min(seq_lengths)
max_length = max(seq_lengths)

# Training loop
def train(model, num_epochs, lr, batch_size, X_train, y_train, seq_lengths):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0  # Track total loss
        num_batches = 0  # Track number of batches

        for i in range(0, len(X_train), batch_size):
            inputs = X_train[i:i + batch_size]
            targets = y_train[i:i + batch_size]
            lengths = seq_lengths[i:i + batch_size]

            optimizer.zero_grad()
            outputs = model(inputs, lengths)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()  # Accumulate total loss
            num_batches += 1  # Count the batch

        avg_loss = epoch_loss / num_batches  # Compute average loss
        train_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    return model, train_losses

def evaluate(model, X_test, y_test, seq_lens):
    model.eval()

    criterion = nn.MSELoss()

    with torch.no_grad():
        outputs = model(X_test, seq_lens)
        loss = criterion(outputs, y_test).item()

    return loss

# initialize and train Vanilla RNN
vanilla_rnn = SequenceModel(input_size, hidden_size, output_size)
print("Training Vanilla RNN:")
vanilla_rnn, vanilla_losses = train(vanilla_rnn, num_epochs, learning_rate, batch_size, X_train, y_train, seq_lengths)

# initialize and train Sequential NN fixing #timesteps to the minimum sequence length
truncated_model = SequenceModelFixedLen(input_size, hidden_size, output_size, seq_len = min_length)
print("\nTraining Truncated Sequence Model:")
truncated_model, truncated_losses = train(truncated_model, num_epochs, learning_rate, batch_size, X_train, y_train, seq_lengths)

# Pad sequences ONLY for the max-length model
X_train_padded, seq_lengths_padded = pad_inputs(X_train, max_length)

# initialize and train Sequential NN fixing #timesteps to the maximum sequence length
# NOTE: it is OK to use torch.nn.utils.rnn.pad_sequence; make sure to set parameter batch_first correctly
padded_model = SequenceModelFixedLen(input_size, hidden_size, output_size, seq_len=max_length)
print("\nTraining Padded Model:")
padded_model, padded_losses = train(padded_model, num_epochs, learning_rate, batch_size, X_train_padded, y_train, seq_lengths_padded)

#Evaluate Models on the Test set
test_seq_lengths = [seq.shape[0] for seq in X_test]
X_test_padded, test_seq_lengths_padded = pad_inputs(X_test, max_length)

vanilla_loss = evaluate(vanilla_rnn, X_test, y_test, test_seq_lengths)
truncated_loss = evaluate(truncated_model, X_test, y_test, test_seq_lengths)
padded_test_loss = evaluate(padded_model, X_test_padded, y_test, test_seq_lengths_padded)

print("\nTest Results:")
print(f"Vanilla RNN Test Loss: {vanilla_loss:.4f}")
print(f"Truncated Model Test Loss: {truncated_loss:.4f}")
print(f"Padded Model Test Loss: {padded_test_loss:.4f}")

plt.figure(figsize=(12,5))
plt.plot(vanilla_losses, label = "Vanilla RNN")
plt.plot(truncated_losses, label = "Truncated Model")
plt.plot(padded_losses, label = "Padded Model")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.show()