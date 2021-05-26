#1D Generative Neural Network for Photon Detection Fast Simulation
from keras import backend as K
from keras import Input
from keras import Model
from keras.utils  import plot_model
from keras.layers import Dense, concatenate, Multiply, Lambda, BatchNormalization, PReLU, ReLU
from keras.optimizers import SGD, Adam

import numpy as np

def outer_product(inputs):
    x, y = inputs
    batchSize = K.shape(x)[0]
    outerProduct = x[:,:, np.newaxis] * y[:, np.newaxis,:]
    outerProduct = K.reshape(outerProduct, (batchSize, -1))
    return outerProduct
    
def vkld_loss(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    diff   = (y_true-y_pred)
    loss   = K.abs(K.sum(diff*K.log(y_pred/y_true), axis=-1))
    return loss

#Network Architecture
#Input: three scalars (Position of scintillation)
#Output: one vecotr (photon detector response)
def model_protodunev7_t0(dim_pdr):
    pos_x       = Input(shape=(1,), name='pos_x')
    pos_y       = Input(shape=(1,), name='pos_y')
    pos_z       = Input(shape=(1,), name='pos_z')
    input_layer = [pos_x, pos_y, pos_z]
        
    feat_npl = Dense(1)(pos_x)
    feat_npl = BatchNormalization(momentum=0.9)(feat_npl)
    feat_npl = ReLU()(feat_npl)
    
    feat_ppl = Dense(1)(pos_x)
    feat_ppl = BatchNormalization(momentum=0.9)(feat_ppl)
    feat_ppl = ReLU()(feat_ppl)
    
    feat_row = Dense(10)(pos_y)
    feat_row = BatchNormalization(momentum=0.9)(feat_row)
    feat_row = ReLU()(feat_row)
    
    feat_col = Dense(3)(pos_z)
    feat_col = BatchNormalization(momentum=0.9)(feat_col)
    feat_col = ReLU()(feat_col)
    
    feat_bar = Lambda(outer_product, output_shape=(30, ))([feat_row, feat_col])
    feat_bar = Dense(30)(feat_bar)
    feat_bar = BatchNormalization(momentum=0.9)(feat_bar)
    feat_bar = ReLU()(feat_bar)
    
    feat_hl1 = Multiply()([feat_bar, feat_npl])    
    feat_hl2 = Multiply()([feat_bar, feat_ppl])
    
    feat_con = concatenate([feat_hl1, feat_hl2])
    feat_con = Dense(dim_pdr)(feat_con)
    feat_con = BatchNormalization(momentum=0.9)(feat_con)
    feat_con = ReLU()(feat_con)
    
    pdr      = Dense(dim_pdr, activation='sigmoid', name='vis_full')(feat_con)
    model    = Model(inputs=input_layer, outputs=pdr, name='protodunev7_model')  
    
    model.summary()
    plot_model(model, to_file='./model_protodunev7.png', show_shapes=True)
    return model    
    
def model_dune10kv4_t0(dim_pdr):
    pos_x       = Input(shape=(1,), name='pos_x')
    pos_y       = Input(shape=(1,), name='pos_y')
    pos_z       = Input(shape=(1,), name='pos_z')
    input_layer = [pos_x, pos_y, pos_z]
    
    feat_int = Dense(1)(pos_x)
    feat_int = BatchNormalization(momentum=0.9)(feat_int)
    feat_int = ReLU()(feat_int)
        
    feat_row = Dense(10)(pos_y)
    feat_row = BatchNormalization(momentum=0.9)(feat_row)
    feat_row = ReLU()(feat_row)
    
    feat_col = Dense(12)(pos_z)
    feat_col = BatchNormalization(momentum=0.9)(feat_col)
    feat_col = ReLU()(feat_col)
    
    feat_cov = Lambda(outer_product, output_shape=(120, ))([feat_row, feat_col])
    feat_cov = Multiply()([feat_cov, feat_int])
    
    feat_cov = Dense(240)(feat_cov)
    feat_cov = BatchNormalization(momentum=0.9)(feat_cov)
    feat_cov = ReLU()(feat_cov)
    
    pdr      = Dense(dim_pdr, activation='sigmoid', name='vis_full')(feat_cov)
    model    = Model(inputs=input_layer, outputs=pdr, name='dune10k_model')
    
    model.summary()
    plot_model(model, to_file='./model_dune10kv4.png', show_shapes=True)
    return model
