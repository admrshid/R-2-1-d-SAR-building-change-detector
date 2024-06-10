from model.getdata import retrieve_filename
from model.getdata import arrange_data
from model.getdata import encodeclass
from model.getdata import train_test_split
from model.getdata import npframegenerator
from model.evaluate import plot_history
from model.evaluate import plot_confusion_matrix
from model.evaluate import calculate_classification_metrics
from model.model_architecture import create_model
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import time
import os
import sys
import numpy as np
import random

def main(path,classes,segment,num_frames,frame_step,metrics):

  size = 32
  m = 0.75
  lr = 0.001

  print(f'running batch size: {size} and momentum: {m} and lr: {lr}')

  start_time = time.time()

  ### SECTION 1: Import data and convert to iterable

  files = retrieve_filename(path,classes,segment)() # creates a list consisting of path vid files
  #print(files)

  class_file = arrange_data(files,classes)() # rearranges the files into a dict with classes as key
  length_change = len(class_file['change'])
  print(f'len change {length_change}')
  length_no_change = len(class_file['no_change'])
  print(f'len no_change {length_no_change}')

  

  # do this to even out the classes
  if segment:
    # Shuffle the 'no_change' list in place
    #random.seed(42)
    temp = class_file['no_change'][:]  # Create a copy of the list to preserve the original
    random.shuffle(temp)
    
    # Slice the shuffled list
    class_file['no_change'] = temp[:length_change]

  encoded_label = encodeclass(class_file) # extracts the classes from the rearranged data (the dict) and encodes it
  print(encoded_label)
  
  train, test = train_test_split(class_file, {'train': 0.8, 'test': 0.2})() # split into train and test as [(class,list of data)]
  # note that the above just splits, doesnt shuffle

  print(f'{test[0][0]} test files: {test[0][1]}')
  print(f'{test[1][0]} test files: {test[1][1]}')

  traindata = npframegenerator(train,encoded_label,num_frames,frame_step,metrics,training=True) # creates an iterable which returns frame set, label. training = True shuffles it
  testdata = npframegenerator(test,encoded_label,num_frames,frame_step,metrics,training=False)

  channels = len(metrics)

  output_signature = (tf.TensorSpec(shape = (None, None, None, channels), dtype = tf.float64), tf.TensorSpec(shape = (), dtype = tf.int16))

  train_ds = tf.data.Dataset.from_generator(traindata, output_signature = output_signature).batch(size)
  test_ds = tf.data.Dataset.from_generator(testdata, output_signature = output_signature).batch(size)

  vidshape = next(traindata())
 
  vid , label = vidshape
  print(label)
  ### SECTION 2: Define model architecture
  
  HEIGHT = vid.shape[1]
  WIDTH = vid.shape[2]

  model = create_model(HEIGHT,WIDTH,channels,m,num_frames) #the second last arg is for BN moving average

  model.compile(loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True), # set loss and type of gradient descent
                optimizer=tf.keras.optimizers.Adam(learning_rate=lr), metrics=['accuracy'])

  model.summary()

  for layer in model.layers:
    if hasattr(layer, 'kernel_initializer'):
        print(f"Layer: {layer.name}")
        print(f"  Weights initializer: {layer.kernel_initializer}")
        print(f"  Biases initializer: {layer.bias_initializer}")

  ### SECTION 3: Training the model

  history = model.fit(x = train_ds, epochs = 30)

  end_time = time.time()

  print(f'total time for data prep and training: {end_time-start_time}\n')

  model.save_weights('./saved_weights/temp_justincase_to_rewind')

  ### SECTION 4: Evaluation

  fig = plot_history(history)
  os.makedirs('./result', exist_ok=True)
  fig.savefig('./result/loss_accuracy_training.png')
  plt.close(fig)

  train_acc = model.evaluate(train_ds, return_dict = True) # I get different results when using .eval, but I'm certain about using .predict (it just does forward pass and then just calc performance measures as usual)
  print(f'train_acc: {train_acc}')

  test_acc = model.evaluate(test_ds, return_dict = True)
  print(f'test_acc: {test_acc}')

  max_train_acc = np.max(history.history['accuracy'])
  print(f'peak_train_acc: {max_train_acc}')

  mean_train_acc = np.mean(history.history['accuracy'])
  print(f'mean_train_acc: {mean_train_acc}')

  def get_actual_predicted_labels(dataset_ori,train_mode,batch_size):

    if train_mode:

      predicted_labels = []

      dataset = tf.data.Dataset.from_generator(dataset_ori, output_signature = output_signature).batch(batch_size)
      actual = [labels for _, labels in dataset_ori()]
      for x, _ in dataset:
        predicted_temp = model(x, training = True)
        predicted_labels.append(predicted_temp)
      actual = tf.stack(actual, axis=0)
      predicted_labels = tf.concat(predicted_labels,axis=0)
      predicted_labels = tf.argmax(predicted_labels, axis=1)
      predicted_labels = tf.cast(predicted_labels, tf.int16)

      return actual, predicted_labels

    else:

      dataset = tf.data.Dataset.from_generator(dataset_ori, output_signature = output_signature).batch(1) #this just tells how much videos you want to predict at once
      actual = [labels for _, labels in dataset_ori()]
      predicted = model.predict(dataset, batch_size=1)

      actual = tf.stack(actual, axis=0)
      predicted = tf.concat(predicted, axis=0)
      predicted = tf.argmax(predicted, axis=1)

      predicted = tf.cast(predicted, tf.int16)

      return actual, predicted

  traindata = npframegenerator(train,encoded_label,num_frames,frame_step,metrics,training=False) # do this to make sure that the labels and predictions align
  actual, predicted = get_actual_predicted_labels(traindata,False,size)
  #actual_train, predicted_train = get_actual_predicted_labels(traindata,True,size)
  #print(predicted)

  fig = plot_confusion_matrix(actual, predicted, encoded_label.keys(), 'Training')
  fig.savefig('./result/confusion_matrix_training.png')  # Save as PNG file
  plt.close(fig)

  #fig = plot_confusion_matrix(actual_train, predicted_train, encoded_label.keys(), 'Training with train mode')
  #fig.savefig('./result/confusion_matrix_training_trainmode.png')  # Save as PNG file
  #plt.close(fig)
  
  actual_1, predicted_1 = get_actual_predicted_labels(testdata,False,size)
  #actual_1_train, predicted_1_train = get_actual_predicted_labels(testdata,True,size)

  fig = plot_confusion_matrix(actual_1, predicted_1, encoded_label.keys(), 'Testing')
  fig.savefig('./result/confusion_matrix_testing.png')  # Save as PNG file
  plt.close(fig)

  #fig = plot_confusion_matrix(actual_1_train, predicted_1_train, encoded_label.keys(), 'Testing with train mode')
  #fig.savefig('./result/confusion_matrix_testing_trainmode.png')  # Save as PNG file
  #plt.close(fig)

  precision, recall, accuracy = calculate_classification_metrics(actual, predicted, encoded_label.keys())
  print(f'train accuracy: {accuracy}')
  print(f'train precision: {precision}')
  print(f'train recall: {recall}\n')

  print(predicted_1)

  precision, recall, accuracy = calculate_classification_metrics(actual_1, predicted_1, encoded_label.keys())
  print(f'test accuracy: {accuracy}')
  print(f'test precision: {precision}')
  print(f'test recall: {recall}\n')

  actual_2, predicted_2 = get_actual_predicted_labels(testdata,False,size)

  print(predicted_2)

  # I did this to verify that it is consistent
  precision, recall, accuracy = calculate_classification_metrics(actual_2, predicted_2, encoded_label.keys())
  print(f'test2 accuracy: {accuracy}')
  print(f'test2 precision: {precision}')
  print(f'test2 recall: {recall}\n')
  

if __name__ == "__main__":

  path = sys.argv[1]
  classes = sys.argv[2]
  segment = sys.argv[3]
  num_frames = sys.argv[4]
  frame_step = sys.argv[5]
  metrics = sys.argv[6]

  classes = classes.split(',')
  if segment == 'Yes':
    segment = True
  else:
    segment = False
  num_frames = int(num_frames)
  frame_step = int(frame_step)
  metrics = metrics.split(',')

  main(path,classes,segment,num_frames,frame_step,metrics)

"""
path = r"D:\Swindon_geotiffs_2\video_dataset"
classes = ['change','no_change']
segment = "Yes"
num_frames = 33
frame_step = 11
metrics = ['VV','VH','VH_COHERENCE','VV_COHERENCE']

main(path,classes,segment,num_frames,frame_step,metrics)
"""