import os
import json
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.pop('TF_CONFIG', None)
if '.' not in sys.path:
  sys.path.insert(0, '..')

import tensorflow as tf
from mnist import *
from multiprocessing import util



def_cluster = {
    "cluster": {
        "worker": ["10.16.0.9:12345", "localhost:12345"]
    },
    "task": {"type": "worker", "index": 1}
}

# Set the TF_CONFIG environment variable
os.environ['TF_CONFIG'] = json.dumps(def_cluster)

per_worker_batch_size = 64
tf_config = json.loads(os.environ['TF_CONFIG'])
num_workers = len(tf_config['cluster']['worker'])
global_batch_size = per_worker_batch_size * num_workers

num_epochs = 10
num_steps_per_epoch=70

# Checkpoint saving and restoring
def _is_chief(task_type, task_id, cluster_spec):
  return (task_type is None
          or task_type == 'chief'
          or (task_type == 'worker'
              and task_id == 0
              and 'chief' not in cluster_spec.as_dict()))

def _get_temp_dir(dirpath, task_id):
  base_dirpath = 'workertemp_' + str(task_id)
  temp_dir = os.path.join(dirpath, base_dirpath)
  tf.io.gfile.makedirs(temp_dir)
  return temp_dir

def write_filepath(filepath, task_type, task_id, cluster_spec):
  dirpath = os.path.dirname(filepath)
  base = os.path.basename(filepath)
  if not _is_chief(task_type, task_id, cluster_spec):
    dirpath = _get_temp_dir(dirpath, task_id)
  return os.path.join(dirpath, base)

#20 断掉，21 开始的时候主从是否可以同步
checkpoint_dir = os.path.join("/tmp/tf-te", 'ckpt')

# Define Strategy
strategy = tf.distribute.MultiWorkerMirroredStrategy()

with strategy.scope():
  # Model building/compiling need to be within `tf.distribute.Strategy.scope`.
  multi_worker_model = build_cnn_model()

  multi_worker_dataset = strategy.distribute_datasets_from_function(
      lambda input_context: dataset_fn(global_batch_size, input_context))
  optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
  train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='train_accuracy')

@tf.function
def train_step(iterator):
  """Training step function."""

  def step_fn(inputs):
    """Per-Replica step function."""
    x, y = inputs
    with tf.GradientTape() as tape:
      predictions = multi_worker_model(x, training=True)
      per_batch_loss = tf.keras.losses.SparseCategoricalCrossentropy(
          from_logits=True,
          reduction=tf.keras.losses.Reduction.NONE)(y, predictions)
      loss = tf.nn.compute_average_loss(
          per_batch_loss, global_batch_size=global_batch_size)

    grads = tape.gradient(loss, multi_worker_model.trainable_variables)
    optimizer.apply_gradients(
        zip(grads, multi_worker_model.trainable_variables))
    train_accuracy.update_state(y, predictions)

    return loss

  per_replica_losses = strategy.run(step_fn, args=(next(iterator),))
  return strategy.reduce(
      tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)




epoch = tf.Variable(
    initial_value=tf.constant(0, dtype=tf.dtypes.int64), name='epoch')
step_in_epoch = tf.Variable(
    initial_value=tf.constant(0, dtype=tf.dtypes.int64),
    name='step_in_epoch')

task_type, task_id, cluster_spec = (strategy.cluster_resolver.task_type,
                                    strategy.cluster_resolver.task_id,
                                    strategy.cluster_resolver.cluster_spec())

checkpoint = tf.train.Checkpoint(
    model=multi_worker_model, epoch=epoch, step_in_epoch=step_in_epoch)
write_checkpoint_dir = write_filepath(checkpoint_dir, task_type, task_id,
                                      cluster_spec)

checkpoint_manager = tf.train.CheckpointManager(
    checkpoint, directory=write_checkpoint_dir, max_to_keep=1)

# Restoring the checkpoint
latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
if latest_checkpoint:
  print('latest_checkpoint')
  checkpoint.restore(latest_checkpoint)

#num_epochs = 10
# Resume our CTL training
while epoch.numpy() < num_epochs:
  print('while loop current epoch is : ', epoch.numpy())
  iterator = iter(multi_worker_dataset)
  total_loss = 0.0
  num_batches = 0
  if step_in_epoch.numpy() == num_steps_per_epoch:
      step_in_epoch.assign(0)
  while step_in_epoch.numpy() < num_steps_per_epoch:
    total_loss += train_step(iterator)
    num_batches += 1
    step_in_epoch.assign_add(1)
    check_point_path = checkpoint_manager.save()
  train_loss = total_loss / num_batches
  print('Epoch: %d, accuracy: %f, train_loss: %f.'
                %(epoch.numpy(), train_accuracy.result(), train_loss))

  train_accuracy.reset_state()

  #check_point_path = checkpoint_manager.save()

  if not _is_chief(task_type, task_id, cluster_spec):
    tf.io.gfile.rmtree(write_checkpoint_dir)

  epoch.assign_add(1)
  step_in_epoch.assign(0)