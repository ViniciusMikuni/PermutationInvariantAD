import numpy as np
import os,re
import tensorflow as tf
from tensorflow import keras
import time
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping

import argparse
import utils
from GSGM_uniform import GSGM
from deepsets_cond import DeepSetsAttClass
from tensorflow.keras.callbacks import ModelCheckpoint
import horovod.tensorflow.keras as hvd

tf.random.set_seed(1233)
#tf.keras.backend.set_floatx('float64')
if __name__ == "__main__":
    hvd.init()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


    parser = argparse.ArgumentParser()    
    parser.add_argument('--config', default='config_AD.json', help='Config file with training parameters')
    parser.add_argument('--data_path', default='/global/cfs/cdirs/m3929/TOP', help='Path containing the training files')
    parser.add_argument('--load', action='store_true', default=False,help='Continue training')
    parser.add_argument('--sup', action='store_true', default=False,help='Train a supervised classifier')
    parser.add_argument('--ll', action='store_true', default=False,help='Run the likelihood training')
    parser.add_argument('--dataset', default='gluon_tagging', help='Which dataset to train')

    flags = parser.parse_args()
    config = utils.LoadJson(flags.config)
    npart = 100
        
    labels = [flags.dataset+'.h5',flags.dataset+'2.h5',flags.dataset+'3.h5']    
    
    data_size,training_data,test_data = utils.DataLoader(flags.data_path,
                                                         labels,
                                                         'gluon_tagging',
                                                         hvd.rank(),hvd.size(),
                                                         use_train=True,
                                                         batch_size=config['BATCH'])

    if flags.sup:
        train_data_bkg, test_data_bkg = utils.DataLoader(flags.data_path,
                                                         ['gluon_tagging.h5'],
                                                         'gluon_tagging',
                                                         hvd.rank(),hvd.size(),
                                                         use_train=True,
                                                         make_tf_data = False,
        )

        train_data_sig, test_data_sig = utils.DataLoader(flags.data_path,
                                                         ['top_tagging.h5'],
                                                         'gluon_tagging',
                                                         hvd.rank(),hvd.size(),
                                                         use_train=True,
                                                         make_tf_data = False,
        )
        
        train_labels = tf.data.Dataset.from_tensor_slices(
            np.concatenate([np.ones(train_data_sig.shape[0]),np.zeros(train_data_bkg.shape[0])],0))
        tf_train = tf.data.Dataset.from_tensor_slices(
            np.concatenate([train_data_sig,train_data_bkg],0))
        training_data = tf.data.Dataset.zip((tf_train,train_labels)).shuffle(train_data_sig.shape[0]).repeat().batch(config['BATCH'])

        test_labels = tf.data.Dataset.from_tensor_slices(
            np.concatenate([np.ones(test_data_sig.shape[0]),np.zeros(test_data_bkg.shape[0])],0))
        tf_test = tf.data.Dataset.from_tensor_slices(
            np.concatenate([test_data_sig,test_data_bkg],0))
        test_data = tf.data.Dataset.zip((tf_test,test_labels)).shuffle(test_data_sig.shape[0]).repeat().batch(config['BATCH'])

        loss="binary_crossentropy"
        
        inputs, outputs = DeepSetsAttClass(
            config['NUM_FEAT'],
            num_heads=1,
            num_transformer = 8,
            projection_dim = 64,
        )
        model = keras.Model(inputs=inputs,outputs=outputs)
        

    else:
        model = GSGM(config=config,npart=npart,particle=flags.dataset,ll_training=flags.ll)
        loss = None

    model_name = config['MODEL_NAME']
    if flags.ll:
        model_name+='_ll'
    elif flags.sup:
        model_name+='_class'

    model_name += '_{}'.format(flags.dataset)
        
    checkpoint_folder = '../checkpoints_{}/checkpoint'.format(model_name)
    if flags.load:
        model.load_weights('{}'.format(checkpoint_folder)).expect_partial()
        
    lr_schedule = tf.keras.experimental.CosineDecay(
        initial_learning_rate=config['LR']*hvd.size(),
        decay_steps=config['MAXEPOCH']*int(data_size*0.8/config['BATCH'])
    )
    opt = tf.keras.optimizers.Adamax(learning_rate=lr_schedule)

    # opt = tf.keras.optimizers.Adam(learning_rate=config['LR']*np.sqrt(hvd.size()))

    
    opt = hvd.DistributedOptimizer(
        opt, average_aggregated_gradients=True)

        
    model.compile(            
        optimizer=opt,
        #run_eagerly=True,
        experimental_run_tf_function=False,
        loss = loss,
        weighted_metrics=[])
    
    callbacks = [
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        hvd.callbacks.MetricAverageCallback(),
        EarlyStopping(patience=100,restore_best_weights=True),
    ]

        
    if hvd.rank()==0:
        checkpoint = ModelCheckpoint(checkpoint_folder,mode='auto',
                                     save_best_only=True,
                                     period=1,save_weights_only=True)
        callbacks.append(checkpoint)
        
    
    history = model.fit(
        training_data,
        epochs=config['MAXEPOCH'],
        callbacks=callbacks,
        steps_per_epoch=int(data_size*0.8/config['BATCH']),
        validation_data=test_data,
        validation_steps=int(data_size*0.1/config['BATCH']),
        verbose=1 if hvd.rank()==0 else 0,
        #steps_per_epoch=1,
    )

    
