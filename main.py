from os.path import exists
from os import mkdir, remove
import tensorflow as tf
from tqdm import tqdm
import numpy as np

from configurations import cfg
from utilizations import split_data
from capsNet import CapsNet


def save_results():
    '''
    <ARGUMENTS>
        # NONE
        
    <OUTPUT>
        # Training Data
          ==> loss_results, train_accuracy_results, valid_accuracy_results
          
        # Testing Data
          ==> test_accuracy_results 
          
    [Save Files - Training]
     - Loss, Accuracies
     1. If there's no result directory, then make it!
     2. If it is training_set, make csv files of loss and accuracies.
     3. If there's past results, delete it. 
     4. Write down the loss and accuracies header on csv files that made it before.
    '''

    # 1 
    if not exists(cfg.results):
        mkdir(cfg.results)

    # 2 ( When Training )
    if cfg.is_training:
        loss = cfg.results + '/loss.csv'
        train_accuracy = cfg.results + '/train_accuracy.csv'
        valid_accuracy = cfg.results + '/valid_accuracy.csv'

    # 3     
        if exists(loss):
            remove(loss)
            
        if exists(train_accuracy):
            remove(train_accuracy)    
        
        if exists(valid_accuracy):
            remove(valid_accuracy)
            
    # 4
        loss_results = open(loss, 'w')
        loss_results.write('step,Loss\n')
        
        train_accuracy_results = open(train_accuracy, 'w')
        train_accuracy_results.write('step,Train_Accuracy\n')
        
        valid_accuracy_results = open(valid_accuracy, 'w')
        valid_accuracy_results.write('step,Valid_Accuracy\n')
        
        return loss_results, train_accuracy_results, valid_accuracy_results
    
    else: # ( When Testing )
        '''
        [Save Files - Testing]
         - Loss, Accuracies
         1. If it is testing_set, make csv files of loss and accuracies.
         2. If there's past results, delete it. 
         3. Write down the loss and accuracies header on csv files that made it before.
        '''

        # 1
        test_accuracy = cfg.results + '/test_accuracy.csv'
        
        # 2
        if exists(test_accuracy):
            remove(test_accuracy)
            
        # 3    
        test_accuracy_results = open(test_accuracy, 'w')
        test_accuracy_results.write('test_accuracy\n')
        
        return test_accuracy_results


def train(model, supervisor, num_label):
    '''
    [ Train the model ]
     1. split data into training dataset, validation dataset
     2. Save results as csv files.
     3. Assign required GPU Memory 
    '''
    # 1
    X_train, y_train, train_batch_num, X_valid, y_valid, valid_batch_num = split_data(cfg.batch_size, is_training=True)
    y = y_valid[:valid_batch_num * cfg.batch_size].reshape((-1, 1))

    # 2
    loss_results, train_accuracy_results, valid_accuracy_results = save_results()
    
    # 3
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    '''
    [Data Training Start]
     1. epoch 50 times
     2. tqdm: Instantly make your loops show a smart progress meter
     3. Write down the training loss and accuracies on csv files that made it before.
     4. Write down the validation loss and accuracies on csv files that made it before.
     5. Save model_epoch_NUM_step in logdir
    '''
    with supervisor.managed_session(config=config) as sess:
        
        print("\nNote: all of results will be saved to directory: " + cfg.results)
        
        # 1
        for epoch in range(cfg.epoch):
            print('Training for epoch ' + str(epoch) + '/' + str(cfg.epoch) + ':')
            
            if supervisor.should_stop():
                print('supervisor stoped!')
                break
            
            # 2 
            for step in tqdm(range(train_batch_num), total=train_batch_num, ncols=70, leave=False, unit='b'):
                start = step * cfg.batch_size
                end = start + cfg.batch_size
                global_step = epoch * train_batch_num + step

                # 3
                if global_step % cfg.train_sum_freq == 0:
                    _, loss, train_accuracy, train_summary = sess.run([model.train_operation, model.total_loss, 
                                                                       model.accuracy, model.train_summary])
                    
                    assert not np.isnan(loss), 'Something wrong! loss is nan...'
                    
                    supervisor.summary_writer.add_summary(train_summary, global_step)
                     
                    loss_results.write(str(global_step) + ',' + str(loss) + "\n")
                    loss_results.flush()
                    train_accuracy_results.write(str(global_step) + ',' + str(train_accuracy / cfg.batch_size) + "\n")
                    train_accuracy_results.flush()
                    
                else:
                    sess.run(model.train_operation)
                
                # 4 
                if cfg.valid_sum_freq != 0 and (global_step) % cfg.valid_sum_freq == 0:
                    
                    valid_accuracy = 0
                    
                    for i in range(valid_batch_num):
                        start = i * cfg.batch_size
                        end = start + cfg.batch_size
                        accuracy = sess.run(model.accuracy, {model.X: X_valid[start:end], model.labels: y_valid[start:end]})
                        valid_accuracy += accuracy
                        
                    valid_accuracy = valid_accuracy / (cfg.batch_size * valid_batch_num)
                    valid_accuracy_results.write(str(global_step) + ',' + str(valid_accuracy) + '\n')
                    valid_accuracy_results.flush()
            # 5 
            if (epoch + 1) % cfg.save_freq == 0:
                supervisor.saver.save(sess, cfg.logdir + '/model_epoch_%04d_step_%02d' % (epoch, global_step))
        
        loss_results.close()
        print('Loss results has been saved to ' + cfg.results + '/loss.csv')        
        
        train_accuracy_results.close()
        print('training accuracy has been saved to ' + cfg.results + '/train_accuracy.csv')        
        
        valid_accuracy_results.close()
        print('Validation accuracy has been saved to ' + cfg.results + '/valid_accuracy.csv')        


def evaluation(model, supervisor, num_label):
    '''
    [ Test the model ]
     1. split testing dataset
     2. Save results as csv files.
    '''    
    # 1
    X_test, y_test, test_batch_num = split_data(cfg.batch_size, is_training=False)
    
    # 2
    test_accuracy_results = save_results()

    '''
    [Data Testing Start]
     1. Latest Model Restore in logdir
     2. tqdm: Instantly make your loops show a smart progress meter
     3. Write down the testing accuracy on csv files that made it before.
    '''
    # 1
    with supervisor.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        supervisor.saver.restore(sess, tf.train.latest_checkpoint(cfg.logdir))
        tf.logging.info('Model restored!')
        
        test_accuracy = 0
        
        # 2
        for i in tqdm(range(test_batch_num), total=test_batch_num, ncols=70, leave=False, unit='b'):
            start = i * cfg.batch_size
            end = start + cfg.batch_size
            accuracy = sess.run(model.accuracy, {model.X: X_test[start:end], model.labels: y_test[start:end]})
            test_accuracy += accuracy
        
        # 3
        test_accuracy = test_accuracy / (cfg.batch_size * test_batch_num)
        test_accuracy_results.write(str(test_accuracy))
        test_accuracy_results.close()
        print('Test accuracy has been saved to ' + cfg.results + '/test_accuracy.csv')


def main(_):
    # Need to write #
    
    tf.logging.info(' Graph Loading... ')
    num_label = 10
    model = CapsNet()
    tf.logging.info(' Graph Loaded ! ')

    sv = tf.train.Supervisor(graph=model.graph, logdir=cfg.logdir, save_model_secs=0)

    if cfg.is_training:
        tf.logging.info(' Start training...')
        train(model, sv, num_label)
        tf.logging.info('Training done')
        
    else:
        evaluation(model, sv, num_label)


if __name__ == "__main__":
    tf.app.run()