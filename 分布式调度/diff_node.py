import json
import os

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy


def get_strategy():
    # 创建MirroredStrategy策略
    # config the cluster and task types
    os.environ["TF_CONFIG"] = json.dumps({
        "cluster": {
            "worker": ["10.16.0.9:12345", "10.16.0.8:12345"]
        },
        "task": {"type": "worker", "index": 0}
    })

    #在10.16.0.8 上需要修改 环境变量为 "task": {"type": "worker", "index": 0}
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    print('设备数量: {}'.format(strategy.num_replicas_in_sync))
    return strategy


def load_data(global_batch_size,strategy):
    # 获取和预处理数据集
    buffer_size = 10000
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size).batch(global_batch_size)

    # 创建分布式数据集
    train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
    return train_dist_dataset


def define_model_and_optimizer():
    model = Sequential([
        Flatten(),
        Dense(units=128, activation='relu'),
        Dense(units=10)
    ])
    optimizer = tf.keras.optimizers.Adam()
    return model, optimizer


def compute_loss(labels, predictions, loss_fn, global_batch_size):
    per_example_loss = loss_fn(labels, predictions)
    return tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)


@tf.function
def train_step(inputs, model, loss_fn, optimizer, global_batch_size):
    images, labels = inputs
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = compute_loss(labels, predictions, loss_fn, global_batch_size)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


def run_training(train_dist_dataset, steps, model, loss_fn, optimizer, global_batch_size,strategy):
    for i, inputs in enumerate(train_dist_dataset):
        if i > steps:
            break
        loss = strategy.run(train_step, args=(inputs, model, loss_fn, optimizer, global_batch_size))
        print('训练步数 {}，损失值：{}'.format(i, loss))


def main():
    # 设定策略
    strategy = get_strategy()

    # 批次大小和选用设备数量
    batch_size_per_replica = 64
    global_batch_size = batch_size_per_replica * strategy.num_replicas_in_sync

    # 加载数据
    train_dist_dataset = load_data(global_batch_size,strategy)

    # 在策略范围内定义模型和优化器
    with strategy.scope():
        model, optimizer = define_model_and_optimizer()
        loss_fn = SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    # 开始训练
    steps = 20
    run_training(train_dist_dataset, steps, model, loss_fn, optimizer, global_batch_size,strategy)


if __name__ == "__main__":
    main()

