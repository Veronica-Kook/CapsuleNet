import os
from os.path import join, exists, splitext
from urllib.request import urlretrieve
import gzip
import shutil
import argparse


# MNIST Data
base_url = "http://yann.lecun.com/exdb/mnist/"
train_imgs = base_url + "train-images-idx3-ubyte.gz"
train_labels = base_url + "train-labels-idx1-ubyte.gz"
test_imgs = base_url + "t10k-images-idx3-ubyte.gz"
test_labels = base_url + "t10k-labels-idx1-ubyte.gz"


def download_zip_file(base_url, user_path):
    '''
    Args:
        base_url: the MNIST data url
        user_path: the user's path to save data
    '''
    filename = base_url.split('/')[-1]
    filepath = join(user_path, filename)

    # If there is no user_path directory then make it!    
    if not exists(user_path):
        os.mkdir(user_path)
        
    unzip = splitext(filepath)[0]

    filepath, _ = urlretrieve(base_url, filepath)
    print('\nSuccessfully Downloaded', filename)
    
    return unzip, filename, filepath


def unzip_file(unzip, filename, filepath):
    '''
    Args:
        unzip: file without extension
        filename: file's name
        filepath: user_path/filename
    '''
    with gzip.open(filepath, 'rb') as f_in, open(unzip, 'wb') as f_out:
        print('\nUnzipping the file ', filename)
        shutil.copyfileobj(f_in, f_out)
        print('\nSuccessfully unzipped the file!')


def down_and_unzip(save_dir):
    '''
    Args:
        save_dir: save MNIST Data to 'save_dir'
    '''
    
    # If there is no save_dir directory then make it!
    if not exists(save_dir):
        os.makedirs(save_dir)
    
    # Only MNIST Data
    train_imgs_out, train_imgs_filename, train_imgs_filepath = download_zip_file(train_imgs, save_dir)
    unzip_file(train_imgs_out, train_imgs_filename, train_imgs_filepath)
    
    train_labels_out, train_labels_filename, train_labels_filepath = download_zip_file(train_labels, save_dir)
    unzip_file(train_labels_out, train_labels_filename, train_labels_filepath)
    
    test_imgs_out, test_imgs_filename, test_imgs_filepath = download_zip_file(test_imgs, save_dir)
    unzip_file(test_imgs_out, test_imgs_filename, test_imgs_filepath)
    
    test_labels_out, test_labels_filename, test_labels_filepath = download_zip_file(test_labels, save_dir)   
    unzip_file(test_labels_out, test_labels_filename, test_labels_filepath)
    
    print("\nEvery file had been downloaded as zipped and unzipped files")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Download MNIST Data automatically for you :) (There's only MNIST Data right now.)") 
    save_dir = join('Data', 'MNIST')
    parser.add_argument("--save_dir", default=save_dir)
    args = parser.parse_args()
    down_and_unzip(args.save_dir)
