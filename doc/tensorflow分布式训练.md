# Tensorflow 分布式调度

## 1.代码

参考：[使用 Keras 和 MultiWorkerMirroredStrategy 的自定义训练循环](https://www.tensorflow.org/tutorials/distribute/multi_worker_with_ctl?hl=zh-cn#%E5%AE%8C%E6%95%B4%E4%BB%A3%E7%A0%81%E4%B8%80%E8%A7%88)

实验环境

硬件：一个chef 一个worker

软件：Python 3.9.9 、CUDA Version 12.3 

## 2.分布式调度实验内容

### 2.1 分布式调度实现

以上实验主要通过设置 TF_CONFIG 环境变量来实现跨节点调度：设置 [TF_CONFIG 环境变量](https://www.tensorflow.org/guide/distributed_training?hl=zh-cn#%E8%AE%BE%E7%BD%AE_tf_config_%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F)

<u>Ps:在主节点启动的时候index=0，从节点启动的时候index=1</u>

在TensorFlow的分布式训练中，`TF_CONFIG`是一个用于配置分布式训练环境的环境变量。它告诉TensorFlow集群的架构以及每个节点的角色和任务。这是实现分布式训练的关键步骤之一。

```python
def_cluster = {
    "cluster": {
        "worker": ["localhost:12345", "localhost:23456"]
    },
    "task": {"type": "worker", "index": 1}
}

# Set the TF_CONFIG environment variable
os.environ['TF_CONFIG'] = json.dumps(def_cluster)
```

在TensorFlow的分布式训练脚本中，可以使用`tf.distribute.MultiWorkerMirroredStrategy`来配置分布式训练环境

```python
strategy = tf.distribute.MultiWorkerMirroredStrategy()
with strategy.scope():
  multi_worker_model = build_cnn_model()

  multi_worker_dataset = strategy.distribute_datasets_from_function(
      lambda input_context: dataset_fn(global_batch_size, input_context))
  optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
  train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='train_accuracy')
```

### 2.2已完成的分布式调度的实验内容

以下对应的实验都是基于代码仓库的main.py 函数和mnist.py函数做修改

1. 单卡的大模型训练

   ```python
   #使用cuda 0 
   os.environ["CUDA_VISIBLE_DEVICES"] = "0"
   
   # 检查TensorFlow是否识别到了GPU
   print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
   per_worker_batch_size = 64
   
   #num_workers 配置为1
   num_workers = 1
   global_batch_size = per_worker_batch_size * num_workers
   
   num_epochs = 10
   num_steps_per_epoch=70
   ```

 2. 单机多卡单进程训练

    主节点12345 进程配置修改

    ```python
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    def_cluster = {
        "cluster": {
            "worker": ["chefIp:12345", "chefIp:23456"]
        },
        "task": {"type": "worker", "index": 0}
    }
    ```

    主节点23456 进程配置修改

    ```python
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    def_cluster = {
        "cluster": {
            "worker": ["chefIp:12345", "chefIp:23456"]
        },
        "task": {"type": "worker", "index": 1}
    }
    ```

 3. 分布式单机多进程训练（依赖tensorflow 分布式调度策略）

    设置cuda 设备

    ```python
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    ```

    主节点12345的进程配置

    ```python
    def_cluster = {
        "cluster": {
            "worker": ["chefIp:12345", "chefIp:23456"]
        }, 
        "task": {"type": "worker", "index": 0}
    }
    
    # Set the TF_CONFIG environment variable
    
    os.environ['TF_CONFIG'] = json.dumps(def_cluster)
    ```

    主节点23456 的进程配置

    ```python
    def_cluster = {
        "cluster": {
            "worker": ["chefIp:12345", "chefIp:23456"]
        },
        "task": {"type": "worker", "index": 1}
    }
    ```

    

 4. 分布式弹性训练, 多机多卡训练(依赖tensorflow 分布式调度策略) [无法热弹]

    主节点TF_CONFIG 设置

    ```python
    def_cluster = {
        "cluster": {
            "worker": ["chefIp:12345", "workerIp:23456"]
        },
        "task": {"type": "worker", "index": 0}
    }
    
    # Set the TF_CONFIG environment variable
    os.environ['TF_CONFIG'] = json.dumps(def_cluster)
    ```

    从节点TF_CONFIG设置

    ```python
    def_cluster = {
        "cluster": {
            "worker": ["chefIp:12345", "workerIp:23456"]
        },
        "task": {"type": "worker", "index": 1}
    }
    ```

 5. 保存checkpoint

    ```python
    #checkpoint 的保存内容可以自定义
    checkpoint = tf.train.Checkpoint(
        model=multi_worker_model, epoch=epoch, step_in_epoch=step_in_epoch)
    write_checkpoint_dir = write_filepath(checkpoint_dir, task_type, task_id,
                                          cluster_spec)
    
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, directory=write_checkpoint_dir, max_to_keep=1)
    
    
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    
    ....
    #训练没有结束的时候，每次都保存一次checkpoint
    while step_in_epoch.numpy() < num_steps_per_epoch:
      total_loss += train_step(iterator)
      num_batches += 1
      step_in_epoch.assign_add(1)
      check_point_path = checkpoint_manager.save()
    ```

 6. 主从节点共享checkpoint 、checkpoint一致性（依赖tensorflow 分布式调度策略）

    由于tensorflow 的分布式训练框架可以实现checkpoint 主从节点共享以及checkpoint一致性。所以在测试tf 分布式调度，主从节点是否共享checkpoint 的方式如下：

    执行到step =7 的时候，断掉chef ,再重新启动master 节点，看代码中的epoch 打印是否从7开始

    ```python
    while epoch.numpy() < num_epochs:
      print('while loop current epoch is : ', epoch.numpy())
      iterator = iter(multi_worker_dataset)
      total_loss = 0.0
      num_batches = 0
    ```

 7. h20 单卡调度、checkpoint 保存

    参考以上单卡调度和checkpoint 保存

8. tensorflow 适配pytorch 的数据集
   参考mint.py 的代码做修改，数据集下载到本地，更换dataset_dir的路径

   ```python
   # 加载自定义数据集函数
   def dataset_fn(global_batch_size, input_context=None):
       dataset_dir = 'root/zjiajia/xxx'
       train_dataset = image_dataset_from_directory(
           os.path.join(dataset_dir, 'train'),
           batch_size=global_batch_size,
           image_size=(32, 32),  # 根据需要调整
           shuffle=True
       )
       validation_dataset = image_dataset_from_directory(
           os.path.join(dataset_dir, 'validation'),
           batch_size=global_batch_size,
           image_size=(32, 32)  # 根据需要调整
       )
   
       # 分发数据集
       if input_context:
           train_dataset = train_dataset.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
           validation_dataset = validation_dataset.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
       
       train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
       validation_dataset = validation_dataset.prefetch(tf.data.experimental.AUTOTUNE)
       
       return train_dataset
   ```



通过一系列的实验，我们验证了TensorFlow在单机和多机、多卡和多进程环境下的分布式调度与训练能力。实验的成功表明，TensorFlow具备良好的扩展性与适应性，能够应对从单设备到跨设备、跨节点的多种场景，并保证了训练过程中的数据一致性和模型的可靠性。

## 3.踩坑记录

### 3.1 问题

tensorflow 分布式训练中无法消费 从节点的问题，参考：[环境变量初始化](https://www.cnblogs.com/zhanxiage1994/p/7989340.html)



### 3.2解决方式

针对以上问题，tensorflow 的官方demo 在安装的时候有写到环境变量的初始化问题

在import tensorflow as tf 之前需要对环境变量初始化

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.pop('TF_CONFIG', None)
if '.' not in sys.path:
  sys.path.insert(0, '.')
```

所以在倒入tf 的时候整体是这样的

```python
import os
import json
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.pop('TF_CONFIG', None)
if '.' not in sys.path:
  sys.path.insert(0, '.')

import tensorflow as tf
from mnist import *
from multiprocessing import util
```

