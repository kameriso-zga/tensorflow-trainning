# tensorflow-elastic-traning

## 实验内容

------

基于原生进行大模型训练。

- 关注于训练过程和checkpoint的保存。

基于原生大模型进行分布式和弹性训练。

- 关注checkpoint保存的过程在业务层的实现。
- 关注主从机之间checkpoint的流转过程的实现。

## 实验结论

------



- tensorflow 框架下，单机单卡可以使用 [tf.distribute.MirroredStrategy](https://www.tensorflow.org/guide/distributed_training?hl=zh-cn#mirroredstrategy)  快速实现分布式训练

- tensorflow 分布式框架：通过设置 [TF_CONFIG 环境变量](https://www.tensorflow.org/guide/distributed_training?hl=zh-cn#%E8%AE%BE%E7%BD%AE_tf_config_%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F) 和tf 分布式框架 API [tf.distribute.MultiWorkerMirroredStrategy](https://www.tensorflow.org/guide/distributed_training?hl=zh-cn#multiworkermirroredstrategy)实现分布式训练，在当前实验环境下，无法实现热重启worker，在master 启动后，指定等待时间，满足一定弹性条件的情况下，会重启所有worker并退出。

- 在训练的过程中，**不论是添加节点还是减少节点以及节点故障，都会让所有节点停止工作**，此时需要脚本自己进行恢复。

  tensorflow 的分布式框架支持checkpoint 的主从同步问题

  checkpoint 需要自己保存`ckpt`，`ckpt`内部需要保存网络参数、训练进度等, 以及其他的业务数据。

  

## 实验实操

------

**【实验1】基础原生简易大模型训练**

单卡训练

- 单卡训练之前实验有做过，直接参考：[TensorFlow图像识别](https://gitlab.bingosoft.net/bingomatrix/example/tensorflow-start/blob/develop/README.md)  TensorFlow-Digit-Recognizer 仓库

多卡训练

- 多卡训练中指定多个gpu,通过[tf.distribute.MirroredStrategy](https://www.tensorflow.org/guide/distributed_training?hl=zh-cn#mirroredstrategy) 实现本地多 GPU 之间进行同步训练。参考mutiple_gpu.py
- tensorflow 的多卡训练和pytorch 有多不同的是，需要使用tf 的分布式调度接口tf.distribute.MirroredStrategy来实现，不单指定多gpu。

**【实验2】分布式的弹性训练**

使用tensorflow官方提供的脚本来做分布式训练，该脚本的具体实现有具体的注释，可参考distribute.py。实现分布式调度最主要的是指定TF_CONFIG 的环境变量参数和调用tf 分布式API[`tf.distribute.MultiWorkerMirroredStrategy`](https://www.tensorflow.org/api_docs/python/tf/distribute/MultiWorkerMirroredStrategy?hl=zh-cn)实现的。多卡的参数指定依然跟单机多卡的形式一样，通过指定环境变量的参数CUDA_VISIBLE_DEVICES 实现。

TF_CONFIG` 是一个 JSON 字符串，包含两个主要部分：`cluster` 和 `task，参数说明如下：

**cluster字段**

`cluster` 字段定义了整个集群的拓扑结构，即包含了所有参与训练的 worker 和 ps 的信息。

- `worker`: 定义了 worker 的地址列表。worker 是实际进行模型训练的节点。
- `ps`: 定义了参数服务器（ps）的地址列表。参数服务器负责存储和更新模型参数。

**task字段**

`task` 字段定义了当前节点在集群中的角色和索引。

- `type`: 当前节点的类型，可以是 `worker` 或 `ps`。决定了当前节点是执行计算还是管理参数。
- `index`: 当前节点在其类型中的索引，从 0 开始。用于唯一标识同类型节点中的某个节点。