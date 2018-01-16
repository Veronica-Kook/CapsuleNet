from os.path import join
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from imageio import imwrite


def split_data(batch_size, is_training=True):
    '''
    <ARGUMENTS>
        # batch_size: # of samples that are going to be propagated through the network
        # is_training: If user wants to train; is_training=True, or test; is_training=False
    
    <OUTPUTS>
        # Training Data
          ==> X_train, y_train, train_batch_num, X_valid, y_valid, valid_batch_num
          
        # Testing Data
          ==> X_test, y_test, test_batch_num
    '''
    
    path = join('Data', 'MNIST')
    
    if is_training:
        '''
         [Training datasets - IMAGES]
          1. train_imgs = the training data
          2. array_train_imgs = make train_imgs to an array
          3. From the paper, they use 60K images for training.
             But, array_train_imgs.reshape((len(array_train_imgs)/(28*28),28,28,1)) 
             cannot reshape array of size into shape (60000,28,28,1).
             (Because the actual size is 60000.02040816326)
             Therefore, I started from the array_train_imgs[16] so that I can make the shape 60000.            
        '''
        train_imgs = open(join(path, 'train-images-idx3-ubyte'))
        array_train_imgs = np.fromfile(file=train_imgs, dtype=np.uint8)
        X_training = array_train_imgs[16:].reshape(int(len(array_train_imgs[16:])/(28*28)), 28, 28, 1).astype(np.float32)
        
        '''
         [Training datasets - LABELS]
          1. train_labels = the training data's labels
          2. array_train_labels = make train_labels to an array
          3. From the paper, they use 60K images for training.
             But, array_train_labels.shape = (60008,)
             Therefore, I started from the array_train_labels[8] so that I can make the shape 60000.
        '''
        train_labels = open(join(path, 'train-labels-idx1-ubyte'))
        array_train_labels = np.fromfile(file=train_labels, dtype=np.uint8)
        y_training = array_train_labels[8:].reshape(int(len(array_train_labels[8:]))).astype(np.int32)

        '''
        [Training datasets - Training and Validation datasets]
         - Divide training datasets to train set(55000), validation set(5000)
         - Why should I need to divide it by 255.? Anyone?? Please help.
        '''
        X_train, X_valid, y_train, y_valid = train_test_split(X_training, y_training, test_size = 5000/60000, train_size = 55000/60000)
        
        X_train = X_train / 255.
        X_valid = X_valid / 255.
        
        '''
        [Set batch numbers]
         - training batch num, validation batch num
        '''
        train_batch_num = len(X_train) // batch_size
        valid_batch_num = len(X_valid) // batch_size

        return X_train, y_train, train_batch_num, X_valid, y_valid, valid_batch_num
    
    else:
        '''
        [Testing datasets - IMAGES]
         1. test_imgs = the testing data
         2. array_test_imgs = make test_imgs to an array
         3. From the paper, they use 10K images for testing.
            But, array_test_imgs.reshape((len(array_test_imgs)/(28*28),28,28,1)) 
            cannot reshape array of size into shape (10000,28,28,1).
            (Because the actual size is 10000.020408163266)
            Therefore, I started from the array_test_imgs[16] so that I can make the shape 10000. 
        '''
        test_imgs = open(join(path, 't10k-images-idx3-ubyte'))
        array_test_imgs = np.fromfile(file=test_imgs, dtype=np.uint8)
        X_test = array_test_imgs[16:].reshape(int(len(array_test_imgs[16:])/(28*28)), 28, 28, 1).astype(np.float)
        
        X_test = X_test / 255.
        
        '''
        [Testing datasets - LABELS]
         1. test_labels = the testing data's labels
         2. array_test_labels = make test_labels to an array
         3. From the paper, they use 10K images for training.
            But, array_train_labels.shape = (10008,)
            Therefore, I started from the array_train_labels[8] so that I can make the shape 10000.
        '''
        test_labels = open(join(path, 't10k-labels-idx1-ubyte'))
        array_test_labels = np.fromfile(file=test_labels, dtype=np.uint8)
        y_test = array_test_labels[8:].reshape((10000)).astype(np.int32)

        '''
        [Set batch numbers]
         - testing batch num
        '''
        test_batch_num = len(X_test) // batch_size
        
        return X_test, y_test, test_batch_num


def tensor_data(batch_size, num_threads):
    '''
    <ARGUMENTS>
        # batch_size: # of samples that going to be propagated through the network
        # num_threads: The number of threads enqueuing tensor_list
    
    <OUTPUTS>
        # (X, y) <-- Tensors
        
    [Transform Data into Tensor]
    1. Split the data into train-set and valid-set
    2. Produces a slice of each Tensor in tensor_list - Turn splitted data into tensor
    3. Shuffle the data
    '''
    X_train, y_train, train_batch_num, X_valid, y_valid, valid_batch_num = split_data(batch_size, is_training=True)    
    
    data_queues = tf.train.slice_input_producer([X_train, y_train])
    
    X, y = tf.train.shuffle_batch(data_queues, num_threads=num_threads,
                                  batch_size=batch_size,
                                  capacity=batch_size * 64,
                                  min_after_dequeue=batch_size * 32,
                                  allow_smaller_final_batch=False)
    return (X, y)


def save_imgs(imgs, size, path):
    '''
    <ARGUMENTS>
        # imgs: [batch_size, image_height, image_width]
        # size: a list with tow int elements, [image_height, image_width]
        # path: the path to save images
        
    <OUPUTS>
        # Write Merged Images as a file.
    '''
    imgs = (imgs + 1.) / 2  # inverse_transform
    return imwrite(path, merge_imgs(imgs, size))


def merge_imgs(images, size):
    '''
    <ARGUMENTS>
        # images: [batch_size, image_height, image_width] ==> inverse_transform
        # size: a list with tow int elements, [image_height, image_width]
        
    <OUTPUTS>
        # Merged Images
    '''
    h, w = images.shape[1], images.shape[2]
    imgs = np.zeros((h * size[0], w * size[1], 3))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        imgs[j * h:j * h + h, i * w:i * w + w, :] = image
        
    return imgs