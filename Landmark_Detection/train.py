import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import os
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.losses import *
from keras import backend as K
from keras.utils import plot_model
import pandas as pd
from all_utils import get_data_generator
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras import callbacks
from model import DenseNet


if __name__ == "__main__":

    #### training ......

    r=120
    c=200
    reduceLearningRate  = 0.5

    df_train_path = 'train_patches_gnd.csv'
    df_val_path ='val_patches_gnd.csv'
    train_image_path = 'train_patch_images/'
    val_image_path ='val_patch_images/'

    df_train = pd.read_csv(df_train_path)
    df_val = pd.read_csv (df_val_path)

    train_idx = np.random.permutation(len(df_train))
    val_idx= np.random.permutation(len(df_val))



    batch_size = 12
    valid_batch_size = 12

    train_gen = get_data_generator(df_train, train_idx, for_training=True,image_path=train_image_path, batch_size=batch_size)
    valid_gen = get_data_generator(df_val, val_idx, for_training=True, image_path=val_image_path, batch_size=valid_batch_size)
    
    if 'outputs' not in os.listdir(os.curdir):
        os.mkdir('outputs')
    
    checkpoint_filepath = 'outputs/' + 'model-{epoch:03d}.h5'
    
    
    if 'model-last_epoch.h5' in os.listdir('outputs/'):
        print ('last model loaded')
        model= load_model('outputs/model-last_epoch.h5')
        
        

    else:
        print('created a new model instead')
        model = DenseNet(input_shape= (r,c,1), dense_blocks=5, dense_layers=-1, growth_rate=8, dropout_rate=0.2,
             bottleneck=True, compression=1.0, weight_decay=1e-4, depth=40)


    # training parameters
    adamOpt = Adam(lr=0.0001)
    model.compile(loss='mean_squared_error', optimizer=adamOpt, metrics=['mae', 'mse'])
    model.summary(line_length=200)


    log_filename = 'outputs/' + 'landmarks' +'_results.csv'

    csv_log = callbacks.CSVLogger(log_filename, separator=',', append=True)

    

    checkpoint = callbacks.ModelCheckpoint(checkpoint_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    
    callbacks_list = [csv_log, checkpoint]
    callbacks_list.append(ReduceLROnPlateau(factor=reduceLearningRate, patience=200,
                                               verbose=True))
    callbacks_list.append(EarlyStopping(verbose=True, patience=200))
    
    callbacks_list.append(TensorBoard(log_dir='outputs/logs'))
    
    
        

    history = model.fit_generator(train_gen,
                        steps_per_epoch=len(train_idx)//batch_size,
                        epochs=1000,
                        callbacks=callbacks_list,
                        validation_data=valid_gen,
                        validation_steps=len(val_idx)//valid_batch_size)
