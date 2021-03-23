import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import copy
from sklearn.externals import joblib
from scipy.stats import norm
from keras import backend as K
from keras.layers import Input, Dense, Lambda, Layer, Add, Multiply, LSTM, RepeatVector
from keras.models import Model, Sequential, load_model
from keras import objectives

# ################################################################
# 데이터 불러오기 + 훈련용 검증용 데이터 나누기
# ################################################################
para = pd.read_csv('Para.csv')

with open('two_d_in_data89.pkl', 'rb') as f:
    all_db_in = pickle.load(f)
with open('two_d_out_data.pkl', 'rb') as f:
    all_db_out = pickle.load(f)

# test = pd.read_csv('12_10020_30_0_0_5_noise.csv')
test = pd.read_csv('../YH_LOCA_CSV/12_10015_30_0_0_5.csv')
creat_noise_data = np.random.normal(1, 0.01, test.shape)
noise_data = test * creat_noise_data
test_in = noise_data[para[(para['0.97_']==0)]['CNS_1']]
test_out = noise_data[para[(para['OUT']==1)]['CNS']]
test_A_in = copy.deepcopy(test_in)
test_A_out = copy.deepcopy(test_out)

# # 고장 주입
# test_in.iloc[300:, 10] = 600
# test_out.iloc[300:, 2] = 600

# 모델링때 생성한 데이터 스케일러값 가져오기
with open('scaler_in89.pkl', 'rb') as f:
    scaler_in = pickle.load(f)

with open('scaler_out.pkl', 'rb') as f:
    scaler_out = pickle.load(f)

# all_db_train = 데이터를 묶을때 이미 스케일됨
# all_db_train.iloc[:, :] = scaler.transform(all_db_train.iloc[:, :].values)
test_in.iloc[:,:] = scaler_in.transform(test_in.iloc[:,:].values)
test_out.iloc[:,:] = scaler_out.transform(test_out.iloc[:,:].values)

input_dim = np.shape(all_db_in)[1]
output_dim = np.shape(all_db_out)[1]
timesteps = 10
batch_size = 1
intermediate_dim = 4
latent_dim = 8
epsilon_std = 1

# ################################################################
# MODELING
# ################################################################
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
vae.compile(optimizer='adam', loss=vae_loss, metrics=['acc', 'cosine_proximity'])
vae.summary()

# 모델링에서 생성한 모델 챙겨오기
vae.load_weights('model_0928_20016.h5')

p_data = vae.predict([all_db_in])
all_db_p_train = pd.DataFrame(p_data)  # 고정

all_db_out = pd.DataFrame(all_db_out)
# all_db_p_train.iloc[:, :] = scaler_out.inverse_transform(all_db_p_train.iloc[:, :].values)
# all_db_out.iloc[:, :] = scaler_out.inverse_transform(all_db_out.iloc[:, :].values)

ylabel = list(para[(para['OUT']==1)]['DESC'])

# ################################################################
# THRESHOLD
# ################################################################
temp_UCL = []
# temp_LCL = []

for i in range(output_dim):
    threshold_ = (all_db_out.iloc[:, i] - all_db_p_train.iloc[:, i])
    print(threshold_.describe())
    UCL = np.mean(threshold_) + 3 * np.std(threshold_)
    # LCL = np.mean(threshold_) - 3 * np.std(threshold_)
    temp_UCL.append(UCL)
    # temp_LCL.append(LCL)
    print('UCL = {}'.format(UCL))
    # print('LCL = {}'.format(LCL))

    # plt.figure()
    # plt.hist(mse)
    # plt.show()

predictions_test = vae.predict(test_in)
p_test = pd.DataFrame(predictions_test, columns=para[(para['OUT']==1)]['CNS'])

p_test_scale = copy.deepcopy(p_test)

p_test.iloc[:, :] = scaler_out.inverse_transform(p_test.iloc[:, :].values)
p_test.to_csv('./in89_Noise_48_25032/result.csv')

test_out_scale = copy.deepcopy(test_out)
test_out.iloc[:, :] = scaler_out.inverse_transform(test_out.iloc[:, :].values)

# ################################################################
#
# ################################################################
name = ['Loop1 Tavg', 'Loop2 Tavg', 'Loop3 Tavg',
        'Steam line#1 flow', 'Steam line#2 flow', 'Steam line#3 flow',
        'Main steam flow',
        'PZR Pressure', 'SG #3 level', 'SG #2 level', 'SG #1 level',
        'SG #3 Pressure', 'SG #2 Pressure', 'SG #1 Pressure',
        'Main steam header pressure']

for i in range(output_dim):
    mse = (test_out_scale.iloc[:,i] - p_test_scale.iloc[:, i])**2

    residual = pd.DataFrame({'Residual': mse})
    a = temp_UCL[i]
    # b = temp_LCL[i]

    fig = plt.figure(constrained_layout=True, figsize=(4, 5))
    gs = fig.add_gridspec(8, 1)
    axs = [fig.add_subplot(gs[0:4, :]),  # 1
           fig.add_subplot(gs[4:8, :]),  # 2
           ]

    axs[0].plot(test_A_out.iloc[:, i], marker='o', markersize=4, color='g', label='Normal')
    axs[0].plot(test_out.iloc[:, i], linewidth=0.5, color='b', label='Faulted')
    axs[0].plot(p_test.iloc[:, i], marker='o', markersize=1, color='r', label='Reconstructed')
    axs[0].legend(loc=1, fontsize=8)
    axs[0].set_ylabel(ylabel[i])
    axs[0].tick_params(labelsize=10)
    axs[0].grid()
    #
    axs[1].plot(residual)
    axs[1].axhline(y=a, color='r')
    # axs[1].axhline(y=b, color='r')
    axs[1].set_ylabel('Reconstruction Error')
    axs[1].set_xlabel('Time(s)')
    axs[1].tick_params(labelsize=10)
    axs[1].grid()
    #
    fig.savefig(fname=f'./in89_Noise_48_25032/{name[i]}.png', dpi=600, facecolor=None)
    plt.show()

for i in range(output_dim):
    test_A_out.iloc[:, i].plot(marker='o', markersize=4, color='g', label='Normal')
    test_out.iloc[:, i].plot(linewidth=0.5, color='b', label='Faulted')
    # p_test.iloc[:, i].plot(marker='o', markersize=1, color='r', label='Estimated')
    p_test.iloc[:, i].plot(marker='o', markersize=1, color='r', label='Estimated')
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel(ylabel[i])

    # mse = (test.iloc[:, i] - p_test.iloc[:, i])
    mse = (test_A_out.iloc[:, i] - p_test.iloc[:, i])
    residual = pd.DataFrame({'Residual': mse})
    residual.plot()
    a = temp_UCL[i]
    # b = temp_LCL[i]
    print(a)
    # print(b)
    plt.axhline(y=a, color='r')
    # plt.axhline(y=b, color='r')
    # plt.show()
