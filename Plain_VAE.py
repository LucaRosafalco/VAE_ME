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

# minimal settings%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#NN hyperparameters
n_points   = 32
batch_size = 8
n_epochs   = 80

#data
data_root  = 'D:\\Luca\\Dati\\'
ID         = 'function_fitting\\'
data_name  = 'Homma_Saltelli_'
case       = '1_1'

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

def vae_l(rec_loss,kl):
    vae_l = rec_loss + kl
    return keras.backend.mean(vae_l)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# definition of a classical VAE %%%%%%%%%%%%%%%%%%%%%%
# model definition
class VAE_plain(keras.models.Model):
    def __init__(self):
        super(VAE_plain,self).__init__()
        self.encoder = keras.Sequential(
            [
                keras.layers.InputLayer(input_shape=n_points, name='input_enc'),
                keras.layers.Dense(8, name='dense1', activation='relu'),
                keras.layers.Dense(8,  name='dense2', activation=None,),
            ]
        )

        self.decoder = keras.Sequential(
            [
                keras.layers.InputLayer(input_shape=4, name='input_dec'),
                keras.layers.Dense(8,name='dense1', activation='relu'),
                keras.layers.Dense(n_points,name='dense2', activation=None),
            ]
        )

    def compile(self,opt,rec_loss,kl_loss,vae_l):
        super(VAE_plain, self).compile()
        self.opt       = opt
        self.rec_loss  = rec_loss
        self.kl_loss   = kl_loss
        self.vae_l     = vae_l

    def call(self,inp):
        encoding = self.encoder(inp)
        z_mean, z_log_var = keras.layers.Lambda(lambda x: tf.split(x,num_or_size_splits=2,axis=1))(encoding)
        batch = keras.backend.shape(z_mean)[0]
        dim   = keras.backend.int_shape(z_mean)[1]
        noise = keras.backend.random_normal(shape=(batch, dim))
        z = z_mean + keras.backend.exp(0.5*z_log_var)*noise
        out = self.decoder(z)
        return z_mean,z_log_var,out

    def train_step(self,inp):
        if isinstance(inp, tuple):
            inp = inp[0]
        with tf.GradientTape(persistent=True) as tape:
            z_mean,z_log_var,out=VAE(inp)
            kl = self.kl_loss(z_mean,z_log_var)
            rec= self.rec_loss(inp,out)
            vae_loss = self.vae_l(rec,kl)
        gradients_vae = tape.gradient(vae_loss,self.trainable_variables)
        self.opt.apply_gradients(zip(gradients_vae,self.trainable_variables))
        return {"kl_loss": kl, "rec_loss": rec}

    def test_step(self,inp):
        if isinstance(inp, tuple):
            inp = inp[0]
        z_mean,z_log_var,out=VAE(inp)
        kl = self.kl_loss(z_mean,z_log_var)
        rec= self.rec_loss(inp,out)
        vae_loss = self.vae_l(rec,kl)
        return {"kl_vld_loss": kl, "rec_vld_loss": rec}
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

# plain VAE training %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
VAE = VAE_plain()

VAE.compile(
    opt=keras.optimizers.Adam(),
    rec_loss=rec_loss,
    kl_loss=kl_loss,
    vae_l=vae_l)

history = VAE.fit(X_tr,epochs=n_epochs,batch_size=batch_size,validation_data=[X_test])
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%