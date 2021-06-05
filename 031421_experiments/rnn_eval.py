#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from rnn_tools.rnn_regression import RNN_Regression
from rnn_tools.rnn_utils import RNN_Config

# make a loader using pytorch because I think its sooo much easier
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import models,transforms
from torch.utils import data
from torch.optim import lr_scheduler
import natsort
import glob


class rnn_state_dataset(data.Dataset):
    
    "The RNN dataset pulls batches of timeseries data"
    
    def __init__(self,filedir,lookback,lookforward,spacetime_lb = 20,skips = 0 ,eval_params=None, predict_current = True, data_sets = None):
        
        self.lb = lookback
        self.lf = lookforward
        self.stlb = spacetime_lb
        self.skips = skips
        self.pred_c = predict_current
        self.label_array,self.lookup = self.read_labels(filedir,data_sets)
        self.encoding_array = self.read_encodings(filedir,data_sets)

        if eval_params is None:
            self.mean,self.stdev = self.normalize_state()
        else:
            mean,std = eval_params
            self.mean,self.stdev = self.normalize_state(mean=mean,stdev=std)

    def __len__(self):
        return(len(self.lookup))
    
    def __getitem__(self, index):
        
        #lookup true index
        idx = self.lookup[index]
        if self.skips+1>self.lb:
            print('Warning: skip step is larger than look back range.')
        
        back_range=np.flip(np.arange(idx,idx-self.lb-1,step=-1-self.skips).astype(int))
        forward_range = np.arange(idx,idx+self.lf+1,step=1).astype(int)
        #print(forward_range)
        #print(back_range)
        input_data = (self.label_array[back_range])[:,7:-36]
        input_encodings = self.encoding_array[back_range]

        input_data = np.hstack((input_data,input_encodings))

        '''
        #timestamp = (self.label_array[back_range])[:,0]
        timestamp = np.arange(input_data.shape[0])
        #input_data = np.hstack((timestamp.reshape((-1,1)),input_data))
        '''

        pred_data = self.label_array[int(idx)][1:4]

        '''
        if self.pred_c:
            pred_data = (self.label_array[forward_range])[:,1:4] #try to predict current force.
        else:
            pred_data = (self.label_array[forward_range])[1:,1:4]
        '''
        force_history = (self.label_array[back_range])[:,1:4]

        return input_data,pred_data.squeeze(),force_history

    def read_encodings(self,label_dir,data_sets):

        file_list = natsort.humansorted(glob.glob(label_dir + "/encode_*.txt"))
        encoding_list = []
        if data_sets is not None:
            for i in data_sets:
                data = np.loadtxt(file_list[i-1])
                encoding_list.append(data)
        else:
            for i in range(len(file_list)):
                data = np.loadtxt(file_list[i])
                encoding_list.append(data)

        encodings = np.concatenate(encoding_list,axis=0)

        return encodings

    def read_labels(self,label_dir,data_sets):
        '''
        Loads all the label data, accounting for excluded sets.
        Generates a look up table to constrain get_item to the right range.
        '''
        
        file_list = natsort.humansorted(glob.glob(label_dir + "/labels_*.txt"))
        label_list = []
        start_index = 0
        lookup= np.empty((0,))
        
        if data_sets is not None:
            for i in data_sets:
                data = np.loadtxt(file_list[i-1],delimiter=",")[self.stlb:,:]
                true_index = np.arange(start_index+self.lb,start_index+data.shape[0]-self.lf)
                label_list.append(data)
                start_index += data.shape[0]
                lookup = np.hstack((lookup,true_index))
        else:
            for i in range(len(file_list)):
                data = np.loadtxt(file_list[i],delimiter=",")[self.stlb:,:]
                true_index = np.arange(start_index+self.lb,start_index+data.shape[0]-self.lf)
                label_list.append(data)
                start_index += data.shape[0]
                lookup = np.hstack((lookup,true_index))
        
        labels = np.concatenate(label_list,axis=0)
        
        return labels,lookup
    
    def normalize_state(self,mean=None,stdev=None):
        '''
        Finds the mean and standard deviation of the dataset and applies
        it all values.
        
        Returns the mean and standard deviation
        '''
        if mean is None and stdev is None:
            mean = np.mean(self.label_array,axis=0)
            stdev = np.std(self.label_array,axis=0)
            
        self.raw_label_array = self.label_array
        self.label_array = (self.label_array-mean)/stdev
        
        return mean,stdev

    def print_lookup(self):
        dataset_index = np.arange(len(self.lookup))
        print(dataset_index)
        print(self.lookup)
        
        return dataset_index,self.lookup


# Learning rate.
learning_rate = 0.0025 #0.001

# Batch size.
BATCH_SIZE = 250

# Hyper-parameters
#  X = [ X_featureVector | X_toolSignals ] = [ f0, ... ,f4095 | Tool Status, x, y, z ]
n_input  = 54+2048    # Dimension of the input vector X (e.g., 4096).
n_steps  = 60       # RNN time steps (e.g., 64).
n_hidden = 256      # Number hidden units.
n_output = 3        # Output dimension (6) -> [ Fx Fy Fz Tx Ty Tz ].
n_layers = 2        # Number of RNN layers.

# Dropout probability value (adjust depending on the task, i.e., inference or training).
dropout_probability_val = np.array([1.0, 1.0],dtype=float)
dropout_probability_train = np.array([0.75, 0.75],dtype=float)

tf.reset_default_graph()

# RNN input (X)
X = tf.placeholder("float", [BATCH_SIZE, n_steps, n_input], name='X_input')

# RNN output (Y)
Y = tf.placeholder("float", [BATCH_SIZE, n_output], name='Y_output')

# Dropout probability.
P_dropout = tf.placeholder("float", [n_layers], name='P_drop')

# Cell type
#                   0                   1               2
CELL_TYPE = ['LSTM-Tensorflow', 'GRU-Tensorflow', 'LSTM-Custom']
cell_type = CELL_TYPE[2]

# Recurrent neural net configuration object
rnn_config = RNN_Config()

rnn_config.summaries_name_scope = 'summaries'
rnn_config.learning_rate  = learning_rate
rnn_config.number_inputs  = n_input
rnn_config.number_outputs = n_output
rnn_config.number_hidden_units = n_hidden
rnn_config.number_steps  = n_steps
rnn_config.number_layers = n_layers
rnn_config.batch_size    = BATCH_SIZE
rnn_config.max_grad_norm = 5
rnn_config.enable_gdl_optimization = True # Set to True if you want GDL enabled in the loss function.

# Cell type (use 'LSTM-Custom' to use the LSTM variant with coupled input-forget gates (CIFG)).
rnn_config.cell_type = cell_type

# LSTM with coupled input-forget gates (CIFG).
rnn_config.custom_cell_type = 'CIFG'

# Create a RNN_Regression object to implement the regression task.
rnn_regression_obj = RNN_Regression(config=rnn_config)

# Boolean variable: If True, feature vectors (extracted from video sequences) will be used as input to the RNN.
rnn_regression_obj.select_input_signals['X-Video'] = True

# Boolean variable: If True, tool data will be used as input to the RNN.
rnn_regression_obj.select_input_signals['X-Tool'] = True

# Scaling factors applied to the input ('X-video', 'X-tool') and output ('Y') data.
rnn_regression_obj.scale_data['X-Video'] = 1
rnn_regression_obj.scale_data['X-Tool'] = 1
rnn_regression_obj.scale_data['Y'] = 1

rnn_regression_obj.step_index = 1
rnn_regression_obj.tolerance  = 0.001

# Builds the RNN computational graph, including input/output placeholders, losses, optimizers, and summaries.
rnn_regression_obj.build_rnn()

# Add ops to save and restore all the variables.
saver = tf.train.Saver(max_to_keep=5)

file_location = "encodings_031621"
train_set = [1,3,5,7,8,10,12,14,15,17,19,21,41,42]
test_dataset = rnn_state_dataset(file_location,lookback=n_steps-1,lookforward=0,skips=0,data_sets=train_set)
test_dataloader = data.DataLoader(test_dataset,batch_size=BATCH_SIZE)

#saver = tf.train.Saver(max_to_keep=5)

test_list =  [4,11,18,
              22,23,
              24,25,
              26,27,
              28,29,
              32,33,
              34,36,
              37,38,39,
              45,46,47]

for dataset_num in test_list:
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  print("restoring TF Session")                    
  sess = tf.Session(config = config)

  #sess.run(tf.global_variables_initializer())
  #saver = tf.train.import_meta_graph('best_model.meta')
  saver.restore(sess, './best_model') 

  print("starting inference...")
  print("Performing inference on {}".format(dataset_num))
  # initialize our dataset
  val_dataset =  rnn_state_dataset(file_location,lookback=n_steps-1,lookforward=0,skips=0,eval_params=(test_dataset.mean,test_dataset.stdev), data_sets = [dataset_num])
  val_dataloader = data.DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=False)

  # initialize our storage variables.]
  predictions = np.empty((0,3))
  labels = np.empty((0,3))

  # initialize our initial state.
  cell_state = (rnn_regression_obj.initial_state[0].eval(session=sess),rnn_regression_obj.initial_state[1].eval(session=sess))
  running_loss = 0.0

  # do the inference
  for i,(x,y,z) in enumerate(val_dataloader,1):
    
    #cell_state = (rnn_regression_obj.reset_state[0].eval(session=sess),rnn_regression_obj.reset_state[1].eval(session=sess))

    x = x.numpy()
    y = y.numpy()

    if x.shape[0] != BATCH_SIZE:
      num_pad = BATCH_SIZE-x.shape[0]
      x = np.concatenate((x,np.zeros((num_pad,n_steps,54+2048))),axis=0)
      y = np.concatenate((y,np.zeros((num_pad,3))),axis=0)

    feed_data_val = {
        rnn_regression_obj.X: x,
        rnn_regression_obj.Y: y,
        rnn_regression_obj.P_dropout: np.array([1.0, 1.0],dtype=float),
        rnn_regression_obj.initial_state: cell_state
        }
     
    estimated_force, cell_state , loss = sess.run([rnn_regression_obj.prediction_op, rnn_regression_obj.final_state,rnn_regression_obj.cost_1],
                                                    feed_dict=feed_data_val)
    # remove padding once the prediction is done
    if x.shape[0] != BATCH_SIZE:
      estimated_force = estimated_force[:-num_pad,:]
      predictions = predictions[:-num_pad,:]
    
    # add the new predictions to the recording variables
    predictions = np.vstack((predictions,estimated_force))
    labels = np.vstack((labels,y))
    # update the running loss with the loss of the batch.
    running_loss += loss
    if i%10 == 0:
      print("Batch: {}, Loss: {:.4f} ".format(i,running_loss/(BATCH_SIZE*i)))
        
  #val_loss = running_loss/(len(val_dataloader.dataset))
  val_loss = running_loss/(BATCH_SIZE*i)
  print("Epoch Val Loss: {:.4f}".format(val_loss))

  # save the labels and predictions
  save_output = np.hstack((predictions,labels))
  save_name = "RNN_preds_" + str(dataset_num) + ".preds"
  print("saving..." +  save_name)
  np.savetxt(save_name,save_output)
  sess.close()


