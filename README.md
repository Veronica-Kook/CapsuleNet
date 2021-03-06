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
[ Simple CNN Model ]
- When a person's face image is the input, then the model is able to extract the features for nose, eyes and mouth each correctly. However, the model will wrongly detect Picasso’s portrait is the input. 


[ CapsNet's Main Idea ]
- The key point of CapsNet is that each neuron contains the likelihood as well as properties of the features - a vector [likelihood, orientation, size]. 


- With [likelihood, orientation, size] information, we can detect changable features in the orientation and size among the nose, eyes and ear.


[ Capsule ]
> A capsule is a group of neurons that not only capture the likelihood but also the parameters of the speific features.


[ Iterative dynamic Routing ]


### 1. Routing Algorithm
![Routing Algorithm](https://user-images.githubusercontent.com/22615736/32276558-82589cee-bedd-11e7-8bb8-cead9ff5640a.png)


> Routing a capsule to the capsule in the layer above based on relevancy is called Routing-by-agreement.

[ Expectation-maximization routing (EM routing) ]


- Group capsules to form a part-whole relationship.


 ==> Role of EM routing: Clustering lower level capsules that produce similar predictions.
 
 
 > A higher level feature (a face) is detected by looking for agreement between votes from the capsules one layer below. We use EM routing to cluster capsules that hvae close proximity of the corresponding votes. 


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


### 2. Instructions of Trying Model
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


> #### 3) Start Train MNIST Training datasets (including Valid datasets).
~~~
$ python main.py
~~~


> #### 4) Start Test MNIST Test datasets
~~~
$ python main.py --is_training=False
~~~


### 3. Instructions of Code ( Either Python file or Jupyter Notebook file )
> #### 1) 1<sup>st</sup> Step
>> ##### [ How to download MNIST data ] 
>> **Read** get_data.py or get_data.ipynb


> #### 2) 2<sup>nd</sup> Step
>> ##### [ How to split data into Train, Valid, Test datasets ] 
>> **Read** utilizations.py or utilizations.ipynb
 

> #### 3) 3<sup>rd</sup> Step
>> ##### [ How to build CapsNet's Layer ] 
>> **Read** capsLayer.py or capsLayer.ipynb
 

> #### 4) 4<sup>th</sup> Step
>> ##### [ How to build CapsNet ] 
>> **Read** capsNet.py or capsNet.ipynb
 
 
> #### 5) 5<sup>th</sup> Step
>> ##### [ How to Start the Program ] 
>> **Read** main.py or main.ipynb
 
 
> #### 6) 6<sup>th</sup> Step
>> ##### [ How to Change Parameters that starts with cfg (=configuration) ] 
>> **Read** configurations.py or configurations.ipynb
