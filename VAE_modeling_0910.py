import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib
from scipy.stats import norm
from keras import backend as K
from keras.layers import Input, Dense, Lambda, Layer, Add, Multiply, LSTM, RepeatVector
from keras.models import Model, Sequential
from keras import objectives
import keras

# 미리 선정된 입력 및 출력 parameter 모아둔 csv 파일 챙김
para = pd.read_csv('new_corr_PARA.csv')

# 데이터 불러오기_ 입력과 출력의 개수가 다르기에 각각 호출해야함
with open('two_d_in_data.pkl', 'rb') as f:
    data_in = pickle.load(f)

with open('two_d_out_data.pkl', 'rb') as f:
    data_out = pickle.load(f)

# 모델 훈련 및 검증용 데이터로 나눔 약 9:1 비율임
train_in = data_in[:87969]
valid_in = data_in[87969:]
train_out = data_out[:87969]
valid_out = data_out[87969:]

# #########################################################################
# VAE-LSTM 모델 생성
# #########################################################################

# ### 2D Input LSTM-VAE
input_dim = np.shape(data_in)[1]
output_dim = np.shape(data_out)[1]
timesteps = 10
batch_size = 1
intermediate_dim = 4
latent_dim = 8
epsilon_std=1

x = Input(shape=(input_dim,))

# LSTM encoding
y = RepeatVector(timesteps)(x)
h = LSTM(intermediate_dim)(y)

# VAE Z layer
z_mean = Dense(latent_dim)(h)
z_log_sigma = Dense(latent_dim)(h)


def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim),
                              mean=0., stddev=epsilon_std)
    return z_mean + z_log_sigma * epsilon


z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])

# decoded LSTM layer
decoder_h = LSTM(intermediate_dim, return_sequences=True)
decoder_mean = LSTM(output_dim, return_sequences=False)

h_decoded = RepeatVector(timesteps)(z)
x_decoded_mean = decoder_h(h_decoded)

# decoded layer
x_decoded_mean = decoder_mean(x_decoded_mean)

# end-to-end autoencoder
vae = Model(x, x_decoded_mean)


def vae_loss(x, x_decoded_mean):
    xent_loss = objectives.mse(x, x_decoded_mean)
    kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma))
    loss = xent_loss + kl_loss
    return loss


vae.compile(optimizer='adam', loss=vae_loss,metrics=['acc', 'cosine_proximity'])
vae.summary()

hist = vae.fit([train_in], [train_out], epochs=300, batch_size=32, validation_data=([valid_in], [valid_out]))

# #########################################################################
# 모델 Weights 저장하기
# #########################################################################
vae.save_weights('model_in99.h5')

his = pd.DataFrame(hist.history)
plt.plot(his['loss'], label='train_loss')
plt.plot(his['val_loss'], label='validation_loss')
plt.legend()
# plt.savefig('./result/loss.png', dpi=600)
plt.show()

# model_in93/250 -32 /loss: 1.5876e-04 - acc: 0.9380 - cosine_proximity: 0.9997 - val_loss: 1.3913e-04 - val_acc: 0.9645 - val_cosine_proximity: 0.9997
# model_in97/250 -32 /loss: 2.7351e-04 - acc: 0.9590 - cosine_proximity: 0.9994 - val_loss: 2.3061e-04 - val_acc: 0.9698 - val_cosine_proximity: 0.9995