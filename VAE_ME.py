#VAE_ME.py

#Editor: Luca Rosafalco
#mail:   luca.rosafalco at polimi.it

# import packages%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import os
import numpy as np
import tensorflow as tf
print(tf.__version__)
import math

from tensorflow import keras
from sklearn.model_selection import train_test_split
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

print('numpy==',np.__version__)
print('sklearn==',np.__version__)
print('tensorflow==',tf.__version__)
print('keras==',np.__version__)

# minimal settings%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#NN hyperparameters
n_points   = 32
batch_size = 16
n_epochs   = 20

#data
data_root  = 'D:\\Luca\\Dati\\'
ID         = 'function_fitting\\'
data_name  = 'Homma_Saltelli_'
case       = '1_1'

#number of models
M = 4
# a uniform initial uniform probability
# between models is assumed

#reconstruction trick parameters
rec_trick = 1e-05 #it allows the computation of the
                  #gradients of the main decoder during
                  #training
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# loss functions definitions%%%%%%%%%%%%%%%%%%%%%%%%%%
def rec_loss(y_true,y_pred):
    mse_loss = keras.losses.mse(y_true,y_pred)#*n_points
    return mse_loss

def kl_loss(z_mean,z_log_var):
    kl = 1 + z_log_var - keras.backend.square(z_mean) - keras.backend.exp(z_log_var) 
    kl = keras.backend.sum(kl,axis=-1)
    kl *= -0.5
    return kl

def MEloss(recM,logpM):
    M = recM.shape[0]
    MEloss = 0
    for m in range(M):
        MEloss = MEloss + recM[m]+logpM[m]
    MEloss /= M
    return MEloss

def vae_l(rec_loss,kl):
    vae_l = rec_loss + kl
    return keras.backend.mean(vae_l)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# definition of a model evidence-based VAE %%%%%%%%%%%

# conditional evidence computation
def ME(M,recM,logpM):
    ME_sum = 0
    for m in range(M):
        ME_sum = ME_sum +  keras.backend.exp(recM[m])*keras.backend.exp(logpM[m])
    guevara = None
    for m in range(M):
        che = logpM[m] + recM[m] - keras.backend.log(ME_sum)
        print(che)
        # if tf.math.exp(che) < 1e-06:
        #     che = tf.math.log(1e-06)
        che = tf.expand_dims(che, axis=0)
        if guevara is None:
            guevara = che
        else:
            guevara = tf.concat([guevara,che],0)
    return guevara

# increment epoch
def incr_epoch(cur_epoch,iter_counter,n_iter_per_epoch):
    if iter_counter == 0:
        iter_counter = n_iter_per_epoch
        cur_epoch = cur_epoch + 1
    return cur_epoch, iter_counter

# decrease iteration counter
def decr_iter(iter_counter):
    iter_counter = iter_counter-1    
    return iter_counter

# model definition
class MEVAE(keras.models.Model):
    def __init__(self,M,rec_trick,n_iter_per_epoch):
        super(MEVAE, self).__init__()
        self.M = M
        self.rec_trick = rec_trick
        self.cur_epoch = keras.backend.variable(value=0)
        self.iter_counter = keras.backend.variable(value=n_iter_per_epoch)
        self.n_iter_per_epoch = n_iter_per_epoch

        logpM = np.zeros(M)
        for m in range(M):
            logpM[m] = tf.math.log(1/M)
        logpM = keras.backend.variable(value=logpM)
        self.logpM = logpM

        self.encoder = keras.Sequential(
            [
                keras.layers.InputLayer(input_shape=n_points, name='input_enc'),
                keras.layers.Dense(8, name='dense1', activation='relu'),
                keras.layers.Dense(8,  name='dense2', activation=None),
            ]
        )

        self.decoder = keras.Sequential(
            [
                keras.layers.InputLayer(input_shape=4, name='input_dec'),
                keras.layers.Dense(8,name='dense1', activation='relu'),
                keras.layers.Dense(n_points,name='dense2', activation=None),
            ]
        )

        self.decoder1 = keras.Sequential(
            [
                keras.layers.InputLayer(input_shape=1, name='input_dec1'),
                keras.layers.Dense(8,name='dense11', activation='relu'),
                keras.layers.Dense(n_points,name='dense21', activation=None),
            ]
        )

        self.decoder2 = keras.Sequential(
            [
                keras.layers.InputLayer(input_shape=2, name='input_dec2'),
                keras.layers.Dense(8,name='dense12', activation='relu'),
                keras.layers.Dense(n_points,name='dense22', activation=None),
            ]
        )

        self.decoder3 = keras.Sequential(
            [
                keras.layers.InputLayer(input_shape=3, name='input_dec3'),
                keras.layers.Dense(8,name='dense13', activation='relu'),
                keras.layers.Dense(n_points,name='dense23', activation=None),
            ]
        )

        self.decoder4 = keras.Sequential(
            [
                keras.layers.InputLayer(input_shape=4, name='input_dec4'),
                keras.layers.Dense(8,name='dense14', activation='relu'),
                keras.layers.Dense(n_points,name='dense24', activation=None),
            ]
        )

    def compile(self,opt,rec_loss,kl_loss,MEloss,vae_l,run_eagerly=True):
        super(MEVAE, self).compile()
        self.opt      = opt
        self.rec_loss = rec_loss
        self.kl_loss  = kl_loss        
        self.ME_loss  = MEloss
        self.vae_l    = vae_l

    def call(self,inp):
        encoding = self.encoder(inp)
        z_mean,z_log_var = keras.layers.Lambda(lambda x: tf.split(x,num_or_size_splits=2,axis=1))(encoding)
        batch = keras.backend.int_shape(z_mean)[0]
        dim   = keras.backend.int_shape(z_mean)[1]
        noise = keras.backend.random_normal(shape=(batch,dim))
        z  = z_mean + keras.backend.exp(0.5*z_log_var)*noise
        z1 = keras.layers.Lambda(lambda x: x[:,0])(z)
        z1 = keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(z1)
        z2 = keras.layers.Lambda(lambda x: x[:,0:2])(z)
        z3 = keras.layers.Lambda(lambda x: x[:,0:3])(z)
        z4 = keras.layers.Lambda(lambda x: x[:,0:4])(z)
        out  = self.decoder(z)
        out1 = self.decoder1(z1)
        out2 = self.decoder2(z2)
        out3 = self.decoder3(z3)
        out4 = self.decoder4(z4)
        return z_mean,z_log_var,out,out1,out2,out3,out4

    def train_step(self,inp):
        if isinstance(inp, tuple):
            inp = inp[0]
        with tf.GradientTape(persistent=True) as tape:
            z_mean,z_log_var,out,out1,out2,out3,out4 = VAE_ME(inp)
            kl   = self.kl_loss(z_mean,z_log_var)

            rec1 = self.rec_loss(inp,out1)
            rec1 = keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(rec1)
            rec2 = self.rec_loss(inp,out2)
            rec2 = keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(rec2)
            rec3 = self.rec_loss(inp,out3)
            rec3 = keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(rec3)
            rec4 = self.rec_loss(inp,out4)
            rec4 = keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(rec4)

            recM = keras.layers.Concatenate()([rec1,rec2,rec3,rec4])

            rec1 = keras.layers.Lambda(lambda x: tf.squeeze(x, axis=1))(rec1)
            rec1 = keras.backend.mean(rec1)
            rec2 = keras.layers.Lambda(lambda x: tf.squeeze(x, axis=1))(rec2)
            rec2 = keras.backend.mean(rec2,axis=0)
            rec3 = keras.layers.Lambda(lambda x: tf.squeeze(x, axis=1))(rec3)
            rec3 = keras.backend.mean(rec3,axis=0)
            rec4 = keras.layers.Lambda(lambda x: tf.squeeze(x, axis=1))(rec4)
            rec4 = keras.backend.mean(rec4,axis=0)
            recM = keras.backend.mean(recM,axis=0)
            
            che = keras.backend.exp(self.logpM)
            keras.backend.print_tensor(che)
            guevara= ME(self.M,recM,self.logpM)
            self.logpM.assign(guevara)

            che = decr_iter(self.iter_counter)
            self.iter_counter.assign(che)
            [che,guevara] = incr_epoch(self.cur_epoch,self.iter_counter,self.n_iter_per_epoch)
            self.cur_epoch.assign(che)
            self.iter_counter.assign(guevara)
            keras.backend.print_tensor(che)

            rec = self.rec_loss(inp,out)
            rec = keras.backend.mean(rec)*self.rec_trick

            ME_l  = self.ME_loss(recM,self.logpM)

            vae_l = keras.backend.mean(kl) + ME_l + rec
        #models M_m
        grads1 = tape.gradient(rec1,self.decoder1.trainable_variables)
        self.opt.apply_gradients(zip(grads1,self.decoder1.trainable_variables))
        grads2 = tape.gradient(rec2,self.decoder2.trainable_variables)        
        self.opt.apply_gradients(zip(grads2, self.decoder2.trainable_variables))
        grads3 = tape.gradient(rec3,self.decoder3.trainable_variables)
        self.opt.apply_gradients(zip(grads3, self.decoder3.trainable_variables))
        grads4 = tape.gradient(rec4,self.decoder4.trainable_variables)
        self.opt.apply_gradients(zip(grads4, self.decoder4.trainable_variables))
        #encoder and decoder
        grads_enc = tape.gradient(vae_l,self.encoder.trainable_variables)
        self.opt.apply_gradients(zip(grads_enc, self.encoder.trainable_variables))
        grads_dec = tape.gradient(vae_l,self.decoder.trainable_variables)
        self.opt.apply_gradients(zip(grads_dec, self.decoder.trainable_variables))
        return {"kl": kl}#{"kl": kl, "rec1": rec1, "rec2": rec2, "rec3": rec3, "rec4":rec4, "ME_loss": ME_l, "cond_evid": che}

    def test_step(self,inp):
        if isinstance(inp, tuple):
            inp = inp[0]
        z_mean,z_log_var,out,out1,out2,out3,out4 = VAE_ME(inp)
        kl  = self.kl_loss(z_mean,z_log_var)
        rec = self.rec_loss(inp,out)
        rec1 = self.rec_loss(inp,out1)
        rec2 = self.rec_loss(inp,out2)
        rec3 = self.rec_loss(inp,out3)
        rec4 = self.rec_loss(inp,out4)
        vae_l = self.vae_l(rec,kl)
        return {"kl_vld": kl, "rec_vld": rec}
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# loading data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data_path = data_root + ID + data_name + case + '.csv'                                             
funct_val = np.genfromtxt(data_path)

funct_val.astype(np.float32)

n_instances = len(funct_val) / n_points
n_instances = int(n_instances)
X = np.zeros((n_instances,n_points))
i0 = 0
for i1 in range(n_instances):
    X[i1,0:n_points] = funct_val[i0:i0+n_points]
    i0 = i0 + n_points

X_tr, X_test = train_test_split(X, test_size=0.2, random_state=5)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# ME VAE training %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n_iter_per_epoch = X_tr.shape[0]/batch_size

VAE_ME = MEVAE(M,rec_trick,n_iter_per_epoch)

VAE_ME.compile(
       opt=keras.optimizers.SGD(),
       rec_loss=rec_loss,
       kl_loss=kl_loss,
       MEloss=MEloss,
       vae_l=vae_l)

history = VAE_ME.fit(X_tr,epochs=n_epochs,batch_size=batch_size,validation_data=[X_test])
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%