from tensorflow.keras.layers import BatchNormalization, Layer, TimeDistributed
from tensorflow.keras.layers import Dense, Input, ReLU, Masking,Conv1D
from tensorflow.keras.models import Model
from tensorflow.keras.activations import swish
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np



def DeepSetsAtt(
        #num_part,
        num_feat,
        time_embedding,
        num_heads=4,
        num_transformer = 4,
        projection_dim=32,
        mask = None,
        fourier_features = False,
):


    inputs = Input((None,num_feat))
    masked_inputs = layers.Masking(mask_value=0.0,name='Mask')(inputs)

    #Include the time information as an additional feature fixed for all particles
    time = layers.Dense(projection_dim,activation=None)(time_embedding)
    time = layers.LeakyReLU(alpha=0.01)(time)
    time = layers.Dense(projection_dim)(time)
    time = layers.LeakyReLU(alpha=0.01)(time)
    
    time = layers.Reshape((1,-1))(time)
    time = tf.tile(time,(1,tf.shape(inputs)[1],1))


    if fourier_features:
        #Gaussian features to the inputs
        nproj = 16
        # emb = tf.math.log(10000.0) / (tf.cast(nproj,tf.float32)- 1.)
        # emb = tf.cast(emb,tf.float32)
        # freq = tf.exp(-emb* tf.range(start=0, limit=nproj, dtype=tf.float32))

        freq = tf.range(start=0, limit=nproj, dtype=tf.float32)
        freq = 2.**(freq) * 2 * np.pi        

        
        
        freq = tf.tile(freq[None, None, :], ( 1, 1,tf.shape(inputs)[-1]))        
        h = tf.repeat(masked_inputs, nproj, axis=-1)
        angle = h*freq
        h = tf.concat([tf.math.sin(angle),tf.math.cos(angle)],-1)


        # h = np.pi*2**7 * masked_inputs
        # h = tf.concat([tf.math.sin(h), tf.math.cos(h)], axis=-1)
        masked_features = layers.Concatenate(-1)([masked_inputs,h])


        
        # freqs = tf.range(6, 8, 1)
        # w = 2.**(tf.cast(freqs, dtype=tf.float32)) * 2 * np.pi        
        # w = tf.tile(w[None,None, :], (1, 1, num_feat))

        
        # # Compute features
        # h = tf.repeat(masked_inputs, 2, axis=-1)
        # h = w * h
        # h = tf.concat([tf.math.sin(h), tf.math.cos(h)], axis=-1)
        
        # masked_features = layers.Concatenate(-1)([masked_inputs,h])
    else:
        masked_features = masked_inputs
        
    masked_features = TimeDistributed(Dense(projection_dim,activation=None))(masked_features)
    #masked_features = masked_inputs

    
    #Use the deepsets implementation with attention, so the model learns the relationship between particles in the event
    concat = layers.Add()([masked_features,time])
    tdd = TimeDistributed(Dense(projection_dim,activation=None))(concat)
    tdd = TimeDistributed(layers.LeakyReLU(alpha=0.01))(tdd)
    encoded_patches = TimeDistributed(Dense(projection_dim))(tdd)

    mask_matrix = tf.matmul(mask,tf.transpose(mask,perm=[0,2,1]))
    
    for _ in range(num_transformer):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        #x1 =encoded_patches

        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim//num_heads,
            dropout=0.1)(x1, x1, attention_mask=tf.cast(mask_matrix,tf.bool))
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])

        
        # Layer normalization 2.
        #time_cond = layers.Dense(projection_dim,activation=None)(time)
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)                
        x3 = layers.Dense(4*projection_dim,activation="gelu")(x3)
        x3 = layers.Dense(projection_dim,activation="gelu")(x3)
        time_cond = layers.Dense(projection_dim,activation="gelu",
                                 kernel_initializer="zeros")(time)
        
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2, time])
        

    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)    
    representation = TimeDistributed(Dense(2*projection_dim,activation=None))(representation)    
    representation =  TimeDistributed(layers.LeakyReLU(alpha=0.01))(representation)
    outputs = TimeDistributed(Dense(num_feat,activation=None,kernel_initializer="zeros"))(representation)
    
    return  inputs, outputs





def Resnet(
        inputs,
        end_dim,
        time_embedding,
        num_embed,
        num_layer = 3,
        mlp_dim=128,
        fourier_features = False,
):

    
    #act = layers.LeakyReLU(alpha=0.01)
    act = swish

    def resnet_dense(input_layer,time_layer,hidden_size,nlayers=2):
        layer = input_layer
        residual = layers.Dense(hidden_size)(layer)
        for _ in range(nlayers):
            time = act(layers.Dense(hidden_size,activation=None,
                                    kernel_initializer="zeros")(time_layer))
            layer=act(layers.Dense(hidden_size,activation=None)(layer))
            layer = layers.Add()([time,layer])
            
            #layer = layers.Dropout(0.1)(layer)
        return layers.Add()([residual,layer])

    time = act(layers.Dense(mlp_dim,activation=None)(time_embedding))
    time = act(layers.Dense(mlp_dim,activation=None)(time))

    if fourier_features:
        nproj=16
        # emb = tf.math.log(10000.0) / (tf.cast(nproj,tf.float32)- 1.)
        # emb = tf.cast(emb,tf.float32)
        # freq = tf.exp(-emb* tf.range(start=0, limit=nproj, dtype=tf.float32))

        freq = tf.range(start=0, limit=nproj, dtype=tf.float32)
        freq = 2.**(freq) * 2 * np.pi        


        freq = tf.tile(freq[None, :], ( 1, tf.shape(inputs)[-1]))        
        h = tf.repeat(inputs, nproj, axis=-1)
        angle = h*freq
        
        #angle = inputs*freq
        h = tf.concat([tf.math.sin(angle),tf.math.cos(angle)],-1)
        concat = layers.Concatenate(-1)([inputs,h])

        
        #Gaussian features to the inputs
        # freqs = tf.range(6, 8, 1)
        
        # w = 2.**(tf.cast(freqs, dtype=tf.float32)) * 2 * np.pi
        # w = tf.tile(w[None, :], (1, tf.shape(inputs)[-1]))
        
        # # Compute features
        # h = tf.repeat(inputs, 2, axis=-1)
        # h = w * h
        # h = tf.concat([tf.math.sin(h), tf.math.cos(h)], axis=-1)

        # concat = layers.Concatenate(-1)([inputs,h])
    else:
        concat = inputs
        
    residual = layers.Dense(mlp_dim)(concat)
    
    layer = residual
    for _ in range(num_layer-1):
        layer =  resnet_dense(layer,time_layer=time,hidden_size=mlp_dim)

    layer = act(layers.Dense(mlp_dim)(residual+layer))
    outputs = layers.Dense(end_dim,kernel_initializer="zeros")(layer)
    
    return outputs



def DeepSetsAttClass(
        num_feat,
        num_heads=4,
        num_transformer = 4,
        projection_dim=32,
):


    inputs = Input((None,num_feat))
    masked_inputs = layers.Masking(mask_value=0.0,name='Mask')(inputs)


    masked_features = masked_inputs        
    masked_features = TimeDistributed(Dense(projection_dim,activation=None))(masked_features)
    
    #Use the deepsets implementation with attention, so the model learns the relationship between particles in the event

    tdd = TimeDistributed(Dense(projection_dim,activation=None))(masked_features)
    tdd = TimeDistributed(layers.LeakyReLU(alpha=0.01))(tdd)
    encoded_patches = TimeDistributed(Dense(projection_dim))(tdd)


    
    for _ in range(num_transformer):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        #x1 =encoded_patches

        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim//num_heads,
            dropout=0.1)(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])

        
        # Layer normalization 2.

        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)                
        x3 = layers.Dense(4*projection_dim,activation="gelu")(x3)
        x3 = layers.Dense(projection_dim,activation="gelu")(x3)
        
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])
        

    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    pooled = tf.reduce_mean(representation,1)
    
    representation = Dense(2*projection_dim,activation=None)(pooled)    
    representation =  layers.LeakyReLU(alpha=0.01)(representation)
    outputs = Dense(1,activation='sigmoid',kernel_initializer="zeros")(representation)
    
    return  inputs, outputs
