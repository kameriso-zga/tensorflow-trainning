import json
import os

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy



#以上测试的结果还是在不同的卡上调度


# Worker 0
os.environ["TF_CONFIG"] = json.dumps({
    "cluster": {
        "worker": ["localhost:12345", "localhost:23456"]
    },
    "task": {"type": "worker", "index": 0}
})

# Worker 1
os.environ["TF_CONFIG"] = json.dumps({
    "cluster": {
        "worker": ["localhost:12345", "localhost:23456"]
    },
    "task": {"type": "worker", "index": 1}
})


def get_strategy():
    # 创建MirroredStrategy策略
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    print('设备数量: {}'.format(strategy.num_replicas_in_sync))
    return strategy


def load_data(global_batch_size, strategy):
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


def run_training(train_dist_dataset, steps, model, loss_fn, optimizer, global_batch_size, strategy,checkpoint_manager):
    for i, inputs in enumerate(train_dist_dataset):
        if i > steps:
            break
        loss = strategy.run(train_step, args=(inputs, model, loss_fn, optimizer, global_batch_size))
        print('训练步数 {}，损失值：{}'.format(i, loss))
        if i % 10 == 0:
            checkpoint_manager.save()
        print('训练步数 {}，损失值：{}'.format(i, loss))


def main():
    # 设定策略
    strategy = get_strategy()

    # 批次大小和选用设备数量
    batch_size_per_replica = 64
    global_batch_size = batch_size_per_replica * strategy.num_replicas_in_sync

    # 加载数据
    train_dist_dataset = load_data(global_batch_size, strategy)

    # 在策略范围内定义模型和优化器
    with strategy.scope():
        model, optimizer = define_model_and_optimizer()
        loss_fn = SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        # 定义checkpoint和checkpoint manager
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
        manager = tf.train.CheckpointManager(checkpoint, './tf_ckpts', max_to_keep=3)

    checkpoint_manager = manager
    # 开始训练
    steps = 20
    run_training(train_dist_dataset, steps, model, loss_fn, optimizer, global_batch_size, strategy,checkpoint_manager)


if __name__ == "__main__":
    main()
