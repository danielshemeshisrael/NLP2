from models import *

def prepare_test():
  # Prepare train and test data
  X_train, y_train, X_test, y_test = prepare_data()
  X_test = []
  # Read lines from the test.untagged file
  with open('test.untagged', 'r') as f:
      lines = f.readlines()

  # Create a list of tuples, where each tuple contains the word, left context, right context, and label
  test_data = []
  for line in lines:
      # Create a list containing the words
      test_line = line.split()
      if len(test_line) > 0:
        test_data.append(test_line)

  # iterate over the items in the list
  for i, word in enumerate(test_data):
    # get the left and right neighbors
    left_neighbor = test_data[i-1][0] if i > 0 else None
    right_neighbor = test_data[i+1][0] if i < len(test_data) - 1 else None
    X_test.append(create_word_vector(word[0], left_neighbor, right_neighbor))
  
  knn_predictions = knn_model(X_train, y_train, X_test, ['O']*len(X_test))
  ff_predictions = ff_model(X_train, y_train, X_test, ['O']*len(X_test))
  # Write predictions to file

  i=0
  with open('comp_211388251_207427428.tagged', 'w') as f:
    with open('test.untagged', 'r') as testfile:
      lines = testfile.readlines()
    # Create a list of tuples, where each tuple contains the word, left context, right context, and label
    for line in lines:
        # Create a list containing the words
        test_line = line.split()
        if len(test_line) > 0:
          f.write(f"{test_data[i][0]}\t{ff_predictions[i]}\n")
          i+=1
        else:
          f.write('\n')
  
prepare_test()