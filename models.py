from gensim import downloader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Download pre-trained word2vec model
WORD_2_VEC_PATH = 'word2vec-google-news-300'
model = downloader.load(WORD_2_VEC_PATH)

# Define a function to create a vector representation of a word and its context
def create_word_vector(word, left_context, right_context):
  # Get the word vector from the pre-trained model
  
  if word not in model.vocab:
    return np.random.rand(900)#word_vec = np.random.rand(300)
  else:
    word_vec = model[word]

  # Get the vectors for the left and right context words
  if left_context not in model.vocab:
    return np.random.rand(900)#left_context_vec = model['Cacophony'] #np.random.rand(300)
  else:
    left_context_vec = model[left_context]
  if right_context not in model.vocab:
    return np.random.rand(900)#right_context_vec = model['Cacophony'] #np.random.rand(300)
  else:
    right_context_vec = model[right_context]

  # Concatenate the word vector, left context vector, and right context vector
  # to create the full vector representation of the word and its context
  return np.concatenate([word_vec, left_context_vec, right_context_vec])

def prepare_data():
  # Read lines from the train.tagged file
  with open('train.tagged', 'r') as f:
      lines = f.readlines()

  # Create a list of tuples, where each tuple contains the word, left context, right context, and label
  train_data = []
  for line in lines:
      # Split the line on the tab character
      split_line = line.strip().split('\t')

      if(len(split_line)<2):
        continue

      # Get the word and label from the split line
      word = split_line[0]
      label = split_line[1]

      # Create a list containing the word and label
      train_data.append([word, label])

  X_train = []
  y_train = []
  X_test = []
  y_test = []

  # iterate over the items in the list
  for i, (word, entity) in enumerate(train_data):
    # get the left and right neighbors
    left_neighbor = train_data[i-1][0] if i > 0 else None
    right_neighbor = train_data[i+1][0] if i < len(train_data) - 1 else None
    X_train.append(create_word_vector(word, left_neighbor, right_neighbor))
    y_train.append(entity)
    
  # Read lines from the dev.tagged file
  with open('dev.tagged', 'r') as f:
      lines = f.readlines()

  # Create a list of tuples, where each tuple contains the word, left context, right context, and label
  test_data = []
  for line in lines:
      # Split the line on the tab character
      split_line = line.strip().split('\t')

      if(len(split_line)<2):
        continue

      # Get the word and label from the split line
      word = split_line[0]
      label = split_line[1]

      # Create a list containing the word and label
      test_data.append([word, label])

  # iterate over the items in the list
  for i, (word, entity) in enumerate(test_data):
    # get the left and right neighbors
    left_neighbor = test_data[i-1][0] if i > 0 else None
    right_neighbor = test_data[i+1][0] if i < len(test_data) - 1 else None
    X_test.append(create_word_vector(word, left_neighbor, right_neighbor))
    y_test.append(entity)
  return X_train, y_train, X_test, y_test
  
def knn_model(X_train, y_train, X_test, y_test):
  #Create a KNN classifier with 5 neighbors
  knn_classifier = KNeighborsClassifier(n_neighbors=5)

  #Fit the model to the training data
  knn_classifier.fit(X_train, y_train)

  #Make predictions on the test data
  predictions = knn_classifier.predict(X_test)

  print("KNN model evaluation")

  #Print the accuracy of the model
  print("Accuracy:", knn_classifier.score(X_test, y_test))

  # Calculate F1 Score
  print("F1 Score:", f1_score([y=='O' for y in y_test],[1]*len(y_test)))
  return predictions
  
def ff_model(X_train, y_train, X_test, y_test):
  # define the neural network
  class Net(nn.Module):
      def __init__(self):
          super(Net, self).__init__()
          self.fc1 = nn.Linear(900, 200)
          self.fc2 = nn.Linear(200, 100)
          self.fc3 = nn.Linear(100, 13)
          
      def forward(self, x):
          x = x.view(-1, 900) # reshape input
          x = F.relu(self.fc1(x))
          x = F.relu(self.fc2(x))
          x = self.fc3(x)
          return x

  # create the model
  network = Net()

  # define the loss function and optimizer
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(network.parameters(), lr=0.005, momentum=0.9)

  # map strings to integers
  entity_dict = dict((entity, i) for i, entity in enumerate(y_train))
  #y_train = [entity_dict.get(entity, -1) for entity in y_train]
  entities = list(set([y_train[i] for i in range(len(y_train))]))
  y_train = [entities.index(y) for y in y_train]

  # input
  X_train = torch.Tensor(X_train)
  y_train = torch.LongTensor(y_train)

  # train the model
  for epoch in range(150):
      # forward pass
      outputs = network(X_train)
      # calculate loss
      loss = criterion(outputs, y_train)
      # backward pass
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      
      # print statistics
      if (epoch+1) % 10 == 0:
          print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, 150, loss.item()))

  # evaluate the model

  # input
  X_test = torch.Tensor(X_test)
  y_test = [entities.index(y) for y in y_test]
  y_test = torch.LongTensor(y_test)
  outputs = network(X_test)
  _, predicted = torch.max(outputs.data, 1)
  correct = (predicted == y_test).sum().item()
  total = y_test.size(0)
  print("FF model evaluation")
  # Calculate accuracy
  print('Accuracy of the network on the {} test entities: {} %'.format(total, 100 * correct / total))
  # Calculate F1 Score
  print("F1 Score:", f1_score([1 if y==entities.index('O') and y==y_test[i] else 0 for i, y in enumerate(predicted)],[1]*len(y_test)))
  return [entities[y] for y in predicted]
  
# Prepare the train and test data
X_train, y_train, X_test, y_test = prepare_data()
knn_predictions = knn_model(X_train, y_train, X_test, y_test)
ff_predictions = ff_model(X_train, y_train, X_test, y_test)