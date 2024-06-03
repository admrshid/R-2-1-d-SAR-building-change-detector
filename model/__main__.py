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
import sys
import time
import os

def main(path,classes,segment,num_frames,frame_step,metrics):

  start_time = time.time()

  ### SECTION 1: Import data and convert to iterable

  files = retrieve_filename(path,classes,segment)() # creates a list consisting of path vid files
  print(files)

  class_file = arrange_data(files,classes)() # rearranges the files into a dict with classes as key
  print(f'class_file')

  encoded_label = encodeclass(class_file) # extracts the classes from the rearranged data (the dict) and encodes it
  print(encoded_label)

  train, test = train_test_split(class_file, {'train': 0.8, 'test': 0.2})() # split into train and test as [(class,list of data)]
  # note that the above just splits, doesnt shuffle

  traindata = npframegenerator(train,encoded_label,num_frames,frame_step,metrics,training=True) # creates an iterable which returns frame set, label. training = True shuffles it
  testdata = npframegenerator(test,encoded_label,num_frames,frame_step,metrics)

  channels = len(metrics)

  output_signature = (tf.TensorSpec(shape = (None, None, None, channels), dtype = tf.float64), tf.TensorSpec(shape = (), dtype = tf.int16))

  train_ds = tf.data.Dataset.from_generator(traindata, output_signature = output_signature).batch(8)
  test_ds = tf.data.Dataset.from_generator(testdata, output_signature = output_signature).batch(8)

  vidshape = next(traindata())
  vid , label = vidshape

  ### SECTION 2: Define model architecture

  HEIGHT = vid.shape[1]
  WIDTH = vid.shape[2]

  model = create_model(HEIGHT,WIDTH,channels)

  model.compile(loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True), # set loss and type of gradient descent
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

  model.summary()

  ### SECTION 3: Training the model

  history = model.fit(x = train_ds, epochs = 30)

  end_time = time.time()

  print(f'total time for data prep and training: {end_time-start_time}')

  model.save_weights('./saved_weights/full_unsegment_lr0.001_swindon_firstversion')

  ### SECTION 4: Evaluation

  fig = plot_history(history)
  os.makedirs('./result', exist_ok=True)
  fig.savefig('./result/loss_accuracy_training.png')
  plt.close(fig)

  model.evaluate(test_ds, return_dict = True)

  def get_actual_predicted_labels(dataset_ori):

    dataset = tf.data.Dataset.from_generator(dataset_ori, output_signature = output_signature).batch(1)
    actual = [labels for _, labels in dataset_ori()]
    predicted = model.predict(dataset)

    actual = tf.stack(actual, axis=0)
    predicted = tf.concat(predicted, axis=0)
    predicted = tf.argmax(predicted, axis=1)

    predicted = tf.cast(predicted, tf.int16)

    return actual, predicted

  traindata = npframegenerator(train,encoded_label,num_frames,frame_step,metrics,training=False) # do this to make sure that the labels and predictions align
  actual, predicted = get_actual_predicted_labels(traindata)
  fig = plot_confusion_matrix(actual, predicted, encoded_label.keys(), 'Training')
  fig.savefig('./result/confusion_matrix_training.png')  # Save as PNG file
  plt.close(fig)

  actual_1, predicted_1 = get_actual_predicted_labels(testdata)
  fig = plot_confusion_matrix(actual_1, predicted_1, encoded_label.keys(), 'Testing')
  fig.savefig('./result/confusion_matrix_testing.png')  # Save as PNG file
  plt.close(fig)

  precision, recall = calculate_classification_metrics(actual, predicted, encoded_label.keys())
  print(f'train precision: {precision}')
  print(f'train recall: {recall}')

  precision, recall = calculate_classification_metrics(actual_1, predicted_1, encoded_label.keys())
  print(f'test precision: {precision}')
  print(f'test recall: {recall}')

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
