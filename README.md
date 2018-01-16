# CapsuleNet - "Dynamic Routing Between Capsules"
---
I could try to build **'CapsNet'** for the first time with **following references**.

**[Concepts]**


A. [Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829)

B. [Understanding Dynamic Routing Between Capsules](https://jhui.github.io/2017/11/03/Dynamic-Routing-Between-Capsules/)

**[Code]**


A. [CapsNet-Keras](https://github.com/XifengGuo/CapsNet-Keras)

B. [CapsNet-Tensorflow](https://github.com/naturomics/CapsNet-Tensorflow)


## Overview
---
### 1. Routing Algorithm
![Routing Algorithm](https://user-images.githubusercontent.com/22615736/32276558-82589cee-bedd-11e7-8bb8-cead9ff5640a.png)


### 2. CapsNet Architecture
#### 2-1) A simple CapsNet with 3 layers
![CapsNet](https://bigsnarf.files.wordpress.com/2017/11/capsnet.png?w=630)


#### 2-2) Decoder structure from the DigitCaps layer
![DigitCaps](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSlWxq8LCyystJhn6NqcQOFnzKXaenzzDKP9EEl3p7md1gbUIzh7w)


#### 2-3) Summary of the Architecture
| Layer Name      | Apply           | Output Shape  |
| :---            | :---            | :---          |
| Image           | Raw image array | git status    |
| ReLu Conv1      | Convolution layer with 9x9 kernels output 256 channels, stride 1, no padding with ReLU | 28x28x1 |
| PrimaryCapsules | Convolution capsule layer with 9x9 kernel output 32x6x6 8-D capsule, stride 2, no padding | 20x20x256 |
| DigiCaps        | Capsule output computed from a *W*<sub>*ij*</sub>(16x8 matrix) between *u*<sub>*i*</sub> and *v*<sub>*j*</sub> (*i* from 1 to 32x6x6 and *j* from 1 to 10).       | 10x16      |
| FC1             | Fully connected with ReLU    | 512    |
| FC2             | Fully connected with ReLU       | 1024      |
| Output Image    | Fully connected with sigmoid     | 784 (28x28)   |


## Action
### 1. Requirements
---
* Python 3.6
* Tensorflow 1.4


### 2. Instructions
---
> #### 1) Download this Repository
>> ##### 1-1) Use 'git'
~~~
 $ git clone https://github.com/Veronica-Kook/CapsuleNet.git

 $ cd CapsuleNet
~~~
>> ##### 1-2) Use ![download ZIP button](https://github.com/Veronica-Kook/CapsuleNet.git)


> #### 2) Download MNIST Datasets.
>> ##### 2-1) Use this code on your Command Prompt Commands(Windows) / Terminal(Mac os, Linux).
~~~
$ python get_data.py
~~~
>> ##### 2-2) Download data from the site : http://yann.lecun.com/exdb/mnist/


> #### 3) 
