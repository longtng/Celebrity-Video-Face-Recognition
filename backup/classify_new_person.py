from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import utils.facenet as facenet
import os
import math
import pickle
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

def main():
    args = parse_args()	
    with tf.Graph().as_default():

        with tf.Session() as sess:
            np.random.seed(666)
            datadir = args.align_dir
            modeldir = args.model_path            
            dataset = facenet.get_dataset(datadir)
            #f_map = "data/embedding/maps.csv"
            paths, labels = facenet.get_image_paths_and_labels(dataset)
            label1 = []
            for i in paths:
                person_name = i.split('/')[-2]
                label1.append(person_name)
            #map_test = pd.read_csv(f_map, header=None)
            #labelnum = []
            #for i in label1:
                #ID = int(map_test.loc[map_test[1]==i, 0].values)
                #labelnum.append(ID) 
            label1 = [ i.replace("_"," ") for i in label1]                                                                           
            print('Number of classes: {}'.format(len(dataset)))
            print('Number of images: {}'.format(len(paths)))
            print('Loading feature extraction model')
            
            facenet.load_model(modeldir)

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            
            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            batch_size = 200
            image_size = 160
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                print('{}/{}'.format(i,nrof_batches_per_epoch))
                start_index = i * batch_size
                end_index = min((i + 1) * batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, image_size)
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)
            print('Testing classifier')
            classifier_filename_exp = os.path.expanduser(args.classifier_filename)
            with open(classifier_filename_exp, 'rb') as infile:
                (model,class_names) = pickle.load(infile)
            labelnum = []
            for i in label1:
                num = class_names.index(i)
                labelnum.append(num)
            print('Loaded classifier model from file "%s"' % classifier_filename_exp)
            print(class_names)
            predictions = model.predict_proba(emb_array)
            print(predictions)	
            best_class_indices = np.argmax(predictions, axis=1)
            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
            for i in range(len(best_class_indices)):
                print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))
            print(best_class_indices)
            print('labelnum')
            print(labelnum)
            report = precision_recall_fscore_support(labelnum,best_class_indices,average='weighted')
            print(report[2])

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
	
    #parser.add_argument("--input_dir", type=str,
                       # help='Path to the data directory containing new person images.',
                       # default = "data/new_person" )
    parser.add_argument("--align_dir", type=str,
						help='Path to the data directory containing aligned of new person images.',
						default ="data/aligned_new_person")
    parser.add_argument('--model_path', type=str,
                        help='Path to embedding model',
                        default="model/20180402-114759.pb")
    #parser.add_argument('--classifier', type=str, choices=['KNN','SVM','RF'],
                        #help='The type of classifier to use.',default='KNN')
    parser.add_argument('classifier_filename',
	                    help='Classifier model file name as a pickle (.pkl) file. ' + 
						'For training this is the output and for classification this is an input.')    
    args = parser.parse_args()
    return args


	
if __name__ == '__main__':
    main()
