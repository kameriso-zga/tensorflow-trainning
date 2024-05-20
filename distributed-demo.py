import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy

# 创建MirroredStrategy策略
strategy = tf.distribute.MirroredStrategy()

#num_replicas_in_sync 可以指定gpu 的使用个数和
#可以指定环境变量，CUDA_visible_deveces="0,1"从而可以影响mirroredStrategy 的num_replicas_in_sync 的值
print('设备数量: {}'.format(strategy.num_replicas_in_sync))

# 批次大小和选用设备数量
buffer_size = 10000
batch_size_per_replica = 64
global_batch_size = batch_size_per_replica * strategy.num_replicas_in_sync

# 获取数据集并进行预处理，x_train 是训练数据集的图像部分，y则是标签部分
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size).batch(global_batch_size)

# 创建分布式数据集
train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)

# 是一个上下文管理器，用于在特定的设备或设备集合范围内定义模型和其他相关操作。在这里，模型、优化器和损失函数都被定义在了 strategy 的范围内，意味着这些定义的操作都会在 MirroredStrategy 环境中并行地执行。
with strategy.scope():
    model = Sequential([
        Flatten(),
        Dense(units=128, activation='relu'),
        Dense(units=10)
    ])

    optimizer = tf.keras.optimizers.Adam() #优化器
    loss_fn = SparseCategoricalCrossentropy(from_logits=True, reduction='none')#损失函数


def compute_loss(labels, predictions):
    per_example_loss = loss_fn(labels, predictions)
    return tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)


# 定义训练步骤
@tf.function
def train_step(inputs):
    images, labels = inputs
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = compute_loss(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


# 开始训练
steps = 20
for i, inputs in enumerate(train_dist_dataset):
    if i > steps:
        break

    loss = strategy.run(train_step, args=(inputs,))
    print('训练步数 {}，损失值：{}'.format(i, loss))