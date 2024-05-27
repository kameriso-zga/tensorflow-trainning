# 从Demo到AI框架测试：TensorFlow图像识别

*版本信息*

*tensorflow  2.16.1*

*cuda  12.3*

*Python 3.9.9*

在这个过程中，我们将会使用 TensorFlow 2.16.1 版本，这是一个开源的端到端机器学习平台，提供了包括神经网络在内的一系列机器学习方法，并支持深度学习和其他复杂计算。我们还会用到 CUDA 12.3 版本，CUDA 是一个由 NVIDIA 提供的并行计算平台和编程模型，可以利用 NVIDIA 的 GPU 加速数据科学工作流程。代码编写的语言则是 Python 3.9.9，这是一个高效、灵活而且广泛使用的编程语言，特别在数据科学和机器学习领域有着极广的应用。

在实施层面，我们将用TensorFlow和CUDA建立起AI框架，并进行一次详细的图像识别测试，整个过程将涵盖从基础Demo到AI框架测试的每个步骤。

完整项目代码请参考 ：https://gitlab.bingosoft.net/bingomatrix/example/aiflow-start

*说明：本文档主要存放python model 和一些实验版本内容*

**<u>关于版本问题：</u>**

官网地址：https://tensorflow.google.cn/install/source?hl=en#gpu

以下2.tensorflow 和cuda 版本测试中，测试了tensorflow 和cuda 版本兼容性，最终结果以官方文档为准tensorflow2.16.1 、cuda12.3

![image-20240329170352709](https://gitlab.bingosoft.net/ccpc/docs/images/raw/develop/imgs/1711703032.png)

## 1.代码及环境准备

### 1.1代码仓库

Volcano 代码仓库地址

```shell
git clone git@gitlab.bingosoft.net:bingomatrix/example/volcano-start.git
```

tensorflow 图像识别代码仓库地址

```shell
git clone git@github.com:toughmuffin12/TensorFlow-Digit-Recognizer.git
```

### 1.2 环境准备

#### python 环境准备

创建python依赖的时候需要使用到Virtualenv，Virtualenv 是一个非常实用的 Python 工具，可以用来创建独立的 Python 运行环境。这在你需要安装多个版本的 Python 或者 Python 的第三方库版本不同的项目时非常有用。每一个 Virtualenv 环境都相当于一个沙盒，在这个环境中安装的包和其他环境是隔离的，环境之间互不干扰。

以下是virtualenv的基本使用

```shell
#安装virtualenv
pip install virtualenv
#创建虚拟环境，venv是虚拟环境的名称 ；/usr/bin/python3 是python 二进制文件位置
virtualenv   --python /usr/bin/python3  venv
#进入虚拟环境
source venv/bin/activate
#python 依赖包设置镜像源
pip config set global.index-url https://pypi.mirrors.ustc.edu.cn/simple
#pip 安装对应的包依赖
pip install xxx #需要什么依赖就下载什么依赖

#生成对应的requirement.txt (如果需要打镜像则需要pip freeze 生成对应的依赖)
pip freeze > requirement.txt
#退出虚拟环境
deactivate
```

- virtualenv参考：https://virtualenv.pypa.io/en/latest/index.html

## 2.tensorflow 和cuda 版本测试

说明：此过程只是为了验证tensorflow2.16.1 和cuda 版本 12.6是否兼容

实验结果：tensorflow2.16.1 和cuda 版本 12.6可以兼容但是依赖于tf-nightly 的python依赖包

结论：官方文档对于版本的要求是tensorflow2.16.1 和cuda 版本 12.3

- 参考tensorflow 官网关于版本信息：[Release TensorFlow 2.16.1 · tensorflow/tensorflow · GitHub](https://github.com/tensorflow/tensorflow/releases/tag/v2.16.1)
- 官网地址：https://tensorflow.google.cn/install/source?hl=en#gpu

### 2.1创建virtualenv虚拟环境

参考1.2.1python 环境准备

Python demo示例子

```python
import tensorflow as tf
import tensorrt as trt
with tf.device("/GPU:0"):
    N = 1200
    shape = (N, N)
while True:
    a = tf.square(tf.random.normal(shape))
    b = tf.square(tf.random.normal(shape))
    tf.matmul(a, b)
```

### 2.2根据提示安装对应依赖

```shell
pip install tensorflow 
pip install kras
pip install tf-nightly #python调用cuda GPU需要安装对应的依赖
...
```

### 2.3 运行demo 查看nivida gpu

```shell
[root@host-10-16-0-8 zjiajia] python3 GPU.py 
2024-03-29 13:59:11.706934: I tensorflow/core/util/port.cc:113] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2024-03-29 13:59:12.844370: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
```

查看nivida gpu 运行情况

```shell
watch -n1 nvidia-smi
```

## 3.单机部署tensorflow demo

由于以上测试tensorflow2.16.1 和cuda 版本 12.6不兼容，根据官方建议，使用tensorflow2.16.1 和cuda 版本 12.3单机部署tensorflow

#### 3.1单机部署创建对应的虚拟环境

下载代码，代码参考：1.1代码仓库tensorflow 图像识别代码仓库地址

```shell
virtualenv   --python /usr/bin/python3  venv
source venv/bin/activate
pip config set global.index-url https://pypi.mirrors.ustc.edu.cn/simple
pip install tensorflow
pip install matplotlib
```

#### 3.2 执行main函数

```shell
(venv) [root@master Digit_Recognition_Tensorflow_Keras]# python3 main.py 
2024-04-02 10:35:43.044530: I tensorflow/core/util/port.cc:113] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2024-04-02 10:35:43.093680: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2024-04-02 10:35:43.939951: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
/root/zjaja/TensorFlow-Digit-Recognizer/Digit_Recognition_Tensorflow_Keras/venv/lib/python3.9/site-packages/keras/src/layers/convolutional/base_conv.py:99: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(
2024-04-02 10:35:45.416207: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1928] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 8264 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 2080 Ti, pci bus id: 0000:d8:00.0, compute capability: 7.5
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1712025345.870067  431105 service.cc:145] XLA service 0x7f81c00044d0 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
I0000 00:00:1712025345.870100  431105 service.cc:153]   StreamExecutor device (0): NVIDIA GeForce RTX 2080 Ti, Compute Capability 7.5
2024-04-02 10:35:45.876098: I tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.cc:268] disabling MLIR crash reproducer, set env var `MLIR_CRASH_REPRODUCER_DIRECTORY` to enable.
2024-04-02 10:35:45.910698: I external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:465] Loaded cuDNN version 8907
I0000 00:00:1712025346.261512  431105 device_compiler.h:188] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step    
[9 8 4 ... 7 7 7]
(10000,)
```

以上输出了执行结果如下，代表执行成功，可修改python 对应的代码，打印输出的结果

```
313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step    
[9 8 4 ... 7 7 7]
(10000,)
```

#### 3.3查看nivida gpu 情况

```shell
#查看nivida gpu 运行情况
watch -n1 nvidia-smi
```

## 4.k8s 上运行运行tensorflow

前提要求需要在k8s集群安装 Nvidia-gpu-operator，如集群已经安装了 Nvidia-gpu-operator可跳过

参考文档：https://gitlab.bingosoft.net/bingomatrix/example/pytorch-start/tree/develop

如何查看集群是否安装Nvidia-gpu-operator

```shell
[root@host-10-16-0-8 ~]# kubectl get pod -A | grep gpu-operator
gpu-operator                gpu-feature-discovery-zs9f6                                         2/2     Running                  0                 10d
gpu-operator                gpu-operator-777f475bc6-qdbkn                                       1/1     Running                  0                 10d
gpu-operator                gpu-operator-node-feature-discovery-gc-6f4d497848-td79d             1/1     Running                  0                 10d
gpu-operator                gpu-operator-node-feature-discovery-master-85ff8d9cc5-pfgxb         1/1     Running                  0                 10d
gpu-operator                gpu-operator-node-feature-discovery-worker-mhfdg                    1/1     Running                  0                 10d
gpu-operator                nvidia-container-toolkit-daemonset-wz29z                            1/1     Running                  0                 10d
gpu-operator                nvidia-cuda-validator-72whk                                         0/1     Completed                0                 10d
gpu-operator                nvidia-dcgm-exporter-mk5dq                                          1/1     Running                  0                 10d
gpu-operator                nvidia-device-plugin-daemonset-hvfxh                                2/2     Running                  0                 10d
gpu-operator                nvidia-operator-validator-8hxwx              
```

### 4.1 生成镜像依赖

下载代码，代码参考：1.1代码仓亏tensorflow 图像识别代码仓库地址

```shell
#安装对应依赖
pip install tensorflow 
...
#pip freeze 生成requirement.txt 
pip freeze > requirement.txt
```

### 4.2编写Dockerfile生成对应镜像

```dockerfile
FROM python:3.9.9
WORKDIR /app
COPY . .
RUN pip config set global.index-url [Simple Index](https://pypi.mirrors.ustc.edu.cn/simple) RUN pip install -r requirements.txt --trusted-host https://pypi.mirrors.ustc.edu.cn/simpled-host
```

```shell
#生成对应镜像
docker build -t registry.bingosoft.net/bingomatrix/base_tf_mnist .
#推送到私有镜像仓库
docker login registry.bingosoft.net
docker push registry.bingosoft.net/bingomatrix/base_tf_mnist
```

### 4.3 数据集挂载

由于TensorFlow-Digit-Recognizer依赖的数据集是minist，所以这个地方采取pvc 挂载minist数据集

```shell
#下载minist 数据集到本地
wget https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
```

创建pv tf-demo-pv.yaml

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: mnt-pvc
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: local-storage
  volumeName: mnist-pv
  resources:
    requests:
      storage: 20Gi
```

创建pvc tf-demo-pvc.yaml

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: mnist-pv
spec:
  storageClassName: manual
  capacity:
    storage: 10Gi
  accessModes:
    - ReadWriteOnce
  hostPath:
    path: "/root/zjiajia/tensorflow-kube-file/mnt"
```

```shell
#创建pv pvc
kubectl apply -f  tf-demo-pv.yaml

kubectl apply -f  tf-demo-pvc.yaml
```

查看是否创建成功

![image-20240329151333086](https://gitlab.bingosoft.net/ccpc/docs/images/raw/develop/imgs/1711696413.png)

### 4.4创建pod

tf-demo-pod.yaml

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: tf-demo 
spec:
  containers:
- name: gpu-container
  image: registry.bingosoft.net/bingomatrix/base_tf_mnist
  imagePullPolicy: IfNotPresent
  command: ["python3"]
  args: ["/app/main.py"] #执行容器内的main.py函数进行图像识别
  resources:
    limits:
      nvidia.com/gpu: 1 #请求节点的GPU资源
  volumeMounts:
  - name: mnt-data
    mountPath: /app/mnt #容器内的数据挂载路径
    volumes:
    - name: mnt-data
      persistentVolumeClaim:
      claimName: tf-data-pvc
```

创建pod

```shell
#创建pod 
kubectl apply -f tf-demo-pod.yaml
#查看是否创建成功
kubectl get pod  -A | grep tf-demo-pod
```

### 4.5查看结果

```shell
#查看pod 日志是否有TensorFlow-Digit-Recognizer的图像识别结果
kubectl logs -f tf-demo-pod
#查看nivida gpu 运行情况
watch -n1 nvidia-smi
```

## 5.使用 volcano 部署 tensorflow

Volcano 是一个开源的 Kubernetes 原生批处理系统，专为高性能、高吞吐量的数据处理和机器学习工作负载设计。它提供了一套丰富的特性来支持复杂的大规模计算工作负载，比如 AI、大数据处理和科学计算。Volcano 通过增强 Kubernetes 现有资源调度能力，使其更适合运行高性能计算（HPC）和 AI 工作负载。

关于volcano资源参考官方文档：https://volcano.sh/zh/docs/queue/

### 5.1镜像准备

```shell
cd volcano-start 
#创建队列
kubectl apply -f newqueue.yaml
#将文件复制到etc目录
cp -r volcano /etc
#基于dockerfile打镜像
docker build -t registry.bingosoft.net/bingomatrix/distributed-tensorflow-volcano:0.1
#登陆
docker login registry.bingosoft.net
#推送
docker push registry.bingosoft.net/bingomatrix/distributed-tensorflow:0.0.1
```

### 5.2部署调度器

调度器进行调度作业,调度yaml 文件：volcano1.yml

```shell
kubectl apply -f volcano1.yml 
```

```yaml
[root@master volcano-start]# cat volcano1.yml 
apiVersion: batch.volcano.sh/v1alpha1
kind: Job
metadata:
  name: tensorflow-distributed-job
spec:
  minAvailable: 5  # 1 chief + 2 workers + 2 ps
  schedulerName: volcano
  plugins:
    env: [] 
    svc: []
  policies:
    - event: PodEvicted
      action: RestartJob
  queue: test
  tasks:
    - replicas: 1
      name: chief
      template:
        spec:
          containers:
            - command: #volcano调度器自动将/etc/volcano目录内的文件挂载到容器内，此yaml负载创建文件会在容器内寻找chief、ps、worker的各个负载service地址，并添加相应的端口（2222），按照容器工作类型界定task_type和task_index，生成容器运行所需的TF_CONFIG
                - sh
                - -c
                - |
                  CHIEF_HOST=`cat /etc/volcano/chief.host | sed 's/$/&:2222/g' | sed 's/^/"/;s/$/"/' | tr "\n" ","`;
                  PS_HOST=`cat /etc/volcano/ps.host | sed 's/$/&:2222/g' | sed 's/^/"/;s/$/"/' | tr "\n" ","`;
                  WORKER_HOST=`cat /etc/volcano/worker.host | sed 's/$/&:2222/g' | sed 's/^/"/;s/$/"/' | tr "\n" ","`;
                  export TF_CONFIG={\"cluster\":{\"chief\":[${CHIEF_HOST}],\"worker\":[${WORKER_HOST}],\"ps\":[${PS_HOST}]},\"task\":{\"type\":\"chief\",\"index\":0},\"environment\":\"cloud\"};
                  echo $TF_CONFIG >> /app/tfconfig.txt
                  python mnist3.py
              image: registry.bingosoft.net/bingomatrix/distributed-tensorflow:0.0.1
              name: tensorflow
              ports:
                - containerPort: 2222
                  name: tfjob-port
              resources:
                limits:
                 nvidia.com/gpu: 1 #请求节点的GPU资源
          restartPolicy: Never
    - replicas: 2
      name: worker
      template:
        spec:
          containers:
            - image: registry.bingosoft.net/bingomatrix/distributed-tensorflow:0.0.1
              name: tensorflow
              command:
                - sh
                - -c
                - |
                  CHIEF_HOST=`cat /etc/volcano/chief.host | sed 's/$/&:2222/g' | sed 's/^/"/;s/$/"/' | tr "\n" ","`;
                  PS_HOST=`cat /etc/volcano/ps.host | sed 's/$/&:2222/g' | sed 's/^/"/;s/$/"/' | tr "\n" ","`;
                  WORKER_HOST=`cat /etc/volcano/worker.host | sed 's/$/&:2222/g' | sed 's/^/"/;s/$/"/' | tr "\n" ","`;
                  export TF_CONFIG={\"cluster\":{\"chief\":[${CHIEF_HOST}],\"worker\":[${WORKER_HOST}],\"ps\":[${PS_HOST}]},\"task\":{\"type\":\"worker\",\"index\":${VK_TASK_INDEX}},\"environment\":\"cloud\"};
                  python mnist3.py
              ports:
                - containerPort: 2222
                  name: tfjob-port
              resources:
                limits:
                 nvidia.com/gpu: 1 #请求节点的GPU资源
          restartPolicy: Never
    - replicas: 2
      name: ps
      template:
        spec:
          containers:
            - image: registry.bingosoft.net/bingomatrix/distributed-tensorflow:0.0.1
              name: tensorflow
              command:
                - sh
                - -c
                - |
                  CHIEF_HOST=`cat /etc/volcano/chief.host | sed 's/$/&:2222/g' | sed 's/^/"/;s/$/"/' | tr "\n" ","`;
                  PS_HOST=`cat /etc/volcano/ps.host | sed 's/$/&:2222/g' | sed 's/^/"/;s/$/"/' | tr "\n" ","`;
                  WORKER_HOST=`cat /etc/volcano/worker.host | sed 's/$/&:2222/g' | sed 's/^/"/;s/$/"/' | tr "\n" ","`;
                  export TF_CONFIG={\"cluster\":{\"chief\":[${CHIEF_HOST}],\"worker\":[${WORKER_HOST}],\"ps\":[${PS_HOST}]},\"task\":{\"type\":\"ps\",\"index\":${VK_TASK_INDEX}},\"environment\":\"cloud\"};
                  python mnist3.py
              ports:
                - containerPort: 2222
                  name: tfjob-port
              resources:
                limits:
                 nvidia.com/gpu: 1 #请求节点的GPU资源
          restartPolicy: Never
```

### 5.3 查看结果

```shell
#pod 结果
[root@master volcano-start]#kubectl get pods
NAME                                  READY   STATUS      RESTARTS   AGE
csi-nfs-test-connection               0/1     Error       0          3d22h
job-test-nginx-0                      0/1     Completed   0          6h6m
tensorflow-distributed-job-chief-0    1/1     Running     0          94m
tensorflow-distributed-job-ps-0       1/1     Running     0          94m
tensorflow-distributed-job-ps-1       1/1     Running     0          94m
tensorflow-distributed-job-worker-0   1/1     Running     0          94m
tensorflow-distributed-job-worker-1   1/1     Running     0          94m
```

```shell
#nivida gpu 结果
[root@master volcano-start]# nvidia-smi
Tue Mar 26 15:39:46 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 545.23.08              Driver Version: 545.23.08    CUDA Version: 12.3     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 2080 Ti     Off | 00000000:D8:00.0 Off |                  N/A |
| 32%   50C    P2              48W / 250W |   1543MiB / 11264MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A   1065593      C   python                                      308MiB |
|    0   N/A  N/A   1065598      C   python                                      308MiB |
|    0   N/A  N/A   1065994      C   python                                      308MiB |
|    0   N/A  N/A   1066283      C   python                                      308MiB |
|    0   N/A  N/A   1066284      C   python                                      308MiB |
+---------------------------------------------------------------------------------------+
```
