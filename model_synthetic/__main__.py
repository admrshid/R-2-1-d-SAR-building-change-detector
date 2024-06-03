from model_synthetic.synthetic_data import synthetic_data
from model_synthetic.synthetic_data import np_synthetic_gen
from model_synthetic.evaluate import plot_confusion_matrix
from model_synthetic.evaluate import calculate_classification_metrics
from model_synthetic.evaluate import plot_history
from model_synthetic.model_architecture import create_model
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import time
import os

def main():

  start_time = time.time()

  ### SECTION 1: Import data and convert to iterable

  labels = synthetic_data()
  train_labels, test_labels = labels.getdata({'train':0.8,'test':0.2})
  print(f'train_labels:{train_labels}')

  traindata = np_synthetic_gen(train_labels,training=True)
  testdata = np_synthetic_gen(test_labels,training=False)

  output_signature = (tf.TensorSpec(shape = (None, None, None, 1), dtype = tf.float64), tf.TensorSpec(shape = (), dtype = tf.int16))

  train_ds = tf.data.Dataset.from_generator(traindata, output_signature = output_signature).batch(8)
  test_ds = tf.data.Dataset.from_generator(testdata, output_signature = output_signature).batch(8)

  vidshape = next(traindata())
  vid , label = vidshape

  ### SECTION 2: Define model architecture

  HEIGHT = vid.shape[1]
  WIDTH = vid.shape[2]

  model = create_model(HEIGHT,WIDTH,1)

  model.compile(loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True), # set loss and type of gradient descent
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=['accuracy'])

  model.summary()

  ### SECTION 3: Training the model

  history = model.fit(x = train_ds, epochs = 8)

  end_time = time.time()

  print(f'total time for data prep and training: {end_time-start_time}')

  model.load_weights('./saved_weights_synthetic/test_1')

  ### SECTION 4: Evaluation

  fig = plot_history(history)
  os.makedirs('./result_synthetic', exist_ok = True)
  fig.savefig('./result_synthetic/loss_accuracy_training.png')
  plt.close(fig)

  model.evaluate(train_ds, return_dict = True)

  def get_actual_predicted_labels(dataset_ori):

    dataset = tf.data.Dataset.from_generator(dataset_ori, output_signature = output_signature).batch(1)
    actual = [labels for _, labels in dataset_ori()]
    predicted = model.predict(dataset)

    actual = tf.stack(actual, axis=0)
    predicted = tf.concat(predicted, axis=0)
    predicted = tf.argmax(predicted, axis=1)

    predicted = tf.cast(predicted, tf.int16)

    return actual, predicted

  traindata = np_synthetic_gen(train_labels,training=False) # need to set random.shuffle off
  actual, predicted = get_actual_predicted_labels(traindata)
  fig = plot_confusion_matrix(actual, predicted, ['no_change','change'], 'Training')
  fig.savefig('./result_synthetic/confusion_matrix_training.png')
  plt.close(fig)

  actual_1, predicted_1 = get_actual_predicted_labels(testdata)
  fig = plot_confusion_matrix(actual_1, predicted_1, ['no_change','change'], 'Testing')
  fig.savefig('./result_synthetic/confusion_matrix_testing.png')
  plt.close(fig)

  precision, recall = calculate_classification_metrics(actual, predicted, ['no_change','change'])
  print(f'train precision: {precision}')
  print(f'train recall: {recall}')

  precision, recall = calculate_classification_metrics(actual_1, predicted_1, ['no_change','change'])
  print(f'test precision: {precision}')
  print(f'test recall: {recall}')

if __name__ == "__main__":

  main()
