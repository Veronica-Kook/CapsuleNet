"""
License: Apache-2.0
Author: Huadong Liao
E-mail: naturomics.liao@gmail.com

[ I changed very little. The base code is Naturomics' code. ]
"""

import tensorflow as tf

from configurations import cfg
from utilizations import tensor_data
from capsLayer import CapsLayer


epsilon = 1e-9


class CapsNet(object):
    '''
    [ CapsuleNet's Main Architecture ]
    '''
    def __init__(self, is_training=True):
        self.graph = tf.Graph()
        
        '''
        [ Make a Graph ]
        '''
        with self.graph.as_default():
            
            if is_training: # When Training
                '''
                [ Get Tensor Data to build Architecture ] <-- ** Main Part **
                '''
                self.X, self.labels = tensor_data(cfg.batch_size, cfg.num_threads)
                self.y = tf.one_hot(self.labels, depth=10, axis=1, dtype=tf.float32)

                self.build_architecture()
                self.loss()
                self._summary()

                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer()
                self.train_operation = self.optimizer.minimize(self.total_loss, global_step=self.global_step) 
                
            else: # When Testing
                self.X = tf.placeholder(tf.float32, shape=(cfg.batch_size, 28, 28, 1))
                self.labels = tf.placeholder(tf.int32, shape=(cfg.batch_size, ))
                self.y = tf.reshape(self.labels, shape=(cfg.batch_size, 10, 1))
                self.build_architecture()

        tf.logging.info('Setting up the main structure')

        
    def build_architecture(self):
        
        '''
        [ 1st Convolution Layer ]
            # Conv1, [batch_size, 20, 20, 256]
        '''
        with tf.variable_scope('Conv1_layer'):
            
            conv1 = tf.contrib.layers.conv2d(self.X, num_outputs=256,
                                             kernel_size=9, stride=1,
                                             padding='VALID')
            
            assert conv1.get_shape() == [cfg.batch_size, 20, 20, 256]

        '''
        [ Primary Capsules Layer]
            # PCL, [batch_size, 1152, 8, 1]
        '''            
        with tf.variable_scope('PrimaryCaps_layer'):
            
            primaryCaps = CapsLayer(num_outputs=32, vec_len=8, with_routing=False, layer_type='CONV')
            caps1 = primaryCaps(conv1, kernel_size=9, stride=2)
            
            assert caps1.get_shape() == [cfg.batch_size, 1152, 8, 1]

        '''
        [ DigitCaps Layer]
            # DGL, [batch_size, 10, 16, 1]
        '''             
        with tf.variable_scope('DigitCaps_layer'):
            digitCaps = CapsLayer(num_outputs=10, vec_len=16, with_routing=True, layer_type='FC')
            self.caps2 = digitCaps(caps1)

        '''
        [ Masking ]
         1. calc ||v_c||, then do softmax(||v_c||) ==> [batch_size, 10, 16, 1] ==> [batch_size, 10, 1, 1]
         2. pick out the index of max softmax val of the 10 caps ==> [batch_size, 10, 1, 1] => [batch_size] (index)
         3. indexing
         4. masking with true label <-- default mode   
        '''
        with tf.variable_scope('Masking') as scope:
            # 1
            self.v_length = tf.sqrt(tf.reduce_sum(tf.square(self.caps2), axis=2, keep_dims=True) + epsilon)
            self.softmax_v = tf.nn.softmax(self.v_length, dim=1)
            assert self.softmax_v.get_shape() == [cfg.batch_size, 10, 1, 1]

            # 2
            self.argmax_idx = tf.to_int32(tf.argmax(self.softmax_v, axis=1))
            assert self.argmax_idx.get_shape() == [cfg.batch_size, 1, 1]
            self.argmax_idx = tf.reshape(self.argmax_idx, shape=(cfg.batch_size, ))

            # 3
            if not cfg.mask_with_y:
                masked_v = []
                for batch_size in range(cfg.batch_size):
                    v = self.caps2[batch_size][self.argmax_idx[batch_size], :]
                    masked_v.append(tf.reshape(v, shape=(1, 1, 16, 1)))

                self.masked_v = tf.concat(masked_v, axis=0)
                assert self.masked_v.get_shape() == [cfg.batch_size, 1, 16, 1]
            # 4
            else:
                # self.masked_v = tf.matmul(tf.squeeze(self.caps2), tf.reshape(self.y, (-1, 10, 1)), transpose_a=True)
                self.masked_v = tf.multiply(tf.squeeze(self.caps2), tf.reshape(self.y, (-1, 10, 1)))
                self.v_length = tf.sqrt(tf.reduce_sum(tf.square(self.caps2), axis=2, keep_dims=True) + epsilon)

        '''
        [ Reconstructe the MNIST images with 3 FC layers ]
            # 1st FC: [batch_size, 1, 16, 1] 
              ==> 2nd FC: [batch_size, 16] 
              ==> 3rd FC: [batch_size, 512]
        '''       
        with tf.variable_scope('Decoder') as scope:
            # 1st
            vector_j = tf.reshape(self.masked_v, shape=(cfg.batch_size, -1))
            fc1 = tf.contrib.layers.fully_connected(vector_j, num_outputs=512)
            assert fc1.get_shape() == [cfg.batch_size, 512]
            
            # 2nd
            fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=1024)
            assert fc2.get_shape() == [cfg.batch_size, 1024]
            
            # 3rd
            self.decoded = tf.contrib.layers.fully_connected(fc2, num_outputs=784, activation_fn=tf.sigmoid)

    def loss(self):
        '''
        [The Margin Loss] ==> [batch_size, 10, 1, 1]
         1. max_l = max(0, m_plus-||v_c||)^2
         2. max_r = max(0, ||v_c||-m_minus)^2
         3. reshape: [batch_size, 10, 1, 1] ==> [batch_size, 10]
         4. calculate L_c
         5. calculate margin_loss
        '''
        
        # 1
        max_l = tf.square(tf.maximum(0., cfg.m_plus - self.v_length))
        
        # 2
        max_r = tf.square(tf.maximum(0., self.v_length - cfg.m_minus))
        assert max_l.get_shape() == [cfg.batch_size, 10, 1, 1]
        
        # 3
        max_l = tf.reshape(max_l, shape=(cfg.batch_size, -1))
        max_r = tf.reshape(max_r, shape=(cfg.batch_size, -1))

        # 4
        T_c = self.y
        L_c = T_c * max_l + cfg.lambda_val * (1 - T_c) * max_r
        
        # 5 
        self.margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))

        '''
        [The Reconstruction Loss]
         1. orgin = Reshape X
         2. squared = (decoded - orgin)^2
         3. reconstruction_error = sum(squared)
        '''
        # 1
        orgin = tf.reshape(self.X, shape=(cfg.batch_size, -1))
        
        # 2
        squared = tf.square(self.decoded - orgin)
        
        # 3
        self.reconstruction_error = tf.reduce_mean(squared)

        '''
        [ Total Loss ]
            # The paper uses sum of squared error as reconstruction error, but we
            # have used reduce_mean in `# 2 The reconstruction loss` to calculate
            # mean squared error. In order to keep in line with the paper,the
            # regularization scale should be 0.0005*784=0.392
        '''
        # 3. Total loss

        self.total_loss = self.margin_loss + cfg.regularization_scale * self.reconstruction_error

    '''
    [ Summary ]
    '''
    def _summary(self):
        train_summary = []
        train_summary.append(tf.summary.scalar('train/margin_loss', self.margin_loss))
        train_summary.append(tf.summary.scalar('train/reconstruction_loss', self.reconstruction_error))
        train_summary.append(tf.summary.scalar('train/total_loss', self.total_loss))
        recon_img = tf.reshape(self.decoded, shape=(cfg.batch_size, 28, 28, 1))
        train_summary.append(tf.summary.image('reconstruction_img', recon_img))
        self.train_summary = tf.summary.merge(train_summary)

        correct_prediction = tf.equal(tf.to_int32(self.labels), self.argmax_idx)
        self.accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
