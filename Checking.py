import pandas as pd
import numpy as np
import pickle
import copy
import glob
import os
import time
from keras import backend as K
from keras.layers import Input, Dense, Lambda, LSTM, RepeatVector
from keras.models import Model
from keras import objectives

def creat_noise(data):
    creat_noise_data = np.random.normal(1, 0.05, data.shape)
    noise_data = data * creat_noise_data
    return noise_data

# ################################################################
# 데이터 불러오기 + 훈련용 검증용 데이터 나누기
# ################################################################
start = time.time()

allFiles = glob.glob("../../../YH_SGTR_CSV/*.csv")

in_scale, out_scale = 'scaler_in.pkl', 'scaler_out.pkl'
in_data_pkl, out_data_pkl = 'two_d_in_data.pkl', 'two_d_out_data.pkl'
model_weight = 'SGTR_model.h5'
save_db, save_db2 = 'All_dict.pkl', 'All_dict2.pkl'

para = pd.read_csv('PARA.csv', na_filter=False)
in_col, out_col = list(para[(para['total'] == '0')]['CNS_']), list(para[(para['OUT'] == '1')]['CNS'])
ylabel = list(para[(para['OUT'] == '1')]['DESC'])

###############################################
allFiles_name = []
for i in range(len(allFiles)):
    allFiles_name.append(allFiles[i][12:-4])
    print(allFiles_name)


output_length = len(out_col)
current = para['CURRENT'][:output_length]
high, low = para['HIGH'][:output_length], para['LOW'][:output_length]
input_num, output_num = para['input'][:output_length], para['output'][:output_length]

with open(in_data_pkl, 'rb') as f:
    all_db_in = pickle.load(f)
with open(out_data_pkl, 'rb') as f:
    all_db_out = pickle.load(f)

with open(in_scale, 'rb') as f:
    scaler_in = pickle.load(f)
with open(out_scale, 'rb') as f:
    scaler_out = pickle.load(f)

input_dim, output_dim = np.shape(all_db_in)[1], np.shape(all_db_out)[1]
timesteps, batch_size, intermediate_dim, latent_dim, epsilon_std = 10, 1, 4, 8, 1

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
vae.load_weights(model_weight)

p_data = vae.predict([all_db_in])
all_db_p_train = pd.DataFrame(p_data)
all_db_out = pd.DataFrame(all_db_out)

# ################################################################
# THRESHOLD
# ################################################################
temp_UCL = []
for i in range(output_dim):
    threshold_ = (all_db_out.iloc[:, i] - all_db_p_train.iloc[:, i])
    UCL = np.mean(threshold_) + 1 * np.std(threshold_)
    temp_UCL.append(UCL)

temp_UCL_pd = pd.DataFrame(temp_UCL)
temp_UCL_pd.to_csv('Threshold_.csv')
print("time :", time.time() - start)

over_threshold, col_ = [], []
save_data_all_scenario, mse_calculation = [], []
for scenario, file_ in zip(range(len(allFiles)), allFiles):
    save_data_one_scenario, mse_calcul = [], []
    for fault, Current, High, Low, Input, Output, Folder in zip(range(len(out_col))
            , current, high, low, input_num, output_num, ylabel):
        save_data, mse_cal = [], []
        for _ in range(4):
            test = pd.read_csv(file_)
            test_in = test[in_col]
            test_in = test_in.apply(creat_noise)
            test_out = test[out_col]
            test_out = test_out.apply(creat_noise)
            test_A_in = copy.deepcopy(test_in)
            test_A_out = copy.deepcopy(test_out)

            MODEL_FOLDER = 'fig'
            # if not os.path.exists(MODEL_FOLDER):
            #     os.mkdir(MODEL_FOLDER)
            MODEL_FOLDER_PATH = f'{MODEL_FOLDER}/{file_[12:-4]}'
            # if not os.path.exists(MODEL_FOLDER_PATH):
            #     os.mkdir(MODEL_FOLDER_PATH)
            # 고장 주입 high
            if _ == 0:
                test_in.iloc[300:, int(Input)] = int(High)
                test_out.iloc[300:, int(Output)] = int(High)
                print(f'scenario: {file_}, HIGH: {High}', test_out.columns[int(Output)])
                MODEL_SAVE_FOLDER_PATH = f'{MODEL_FOLDER_PATH}/{Folder}_HIGH_{High}'
                # if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
                #     os.mkdir(MODEL_SAVE_FOLDER_PATH)
            elif _ == 1:
                if Low == '':
                    pass
                else:
                    test_in.iloc[300:, int(Input)] = int(Low)
                    test_out.iloc[300:, int(Output)] = int(Low)
                    print(f'scenario: {file_}, LOW: {Low}', test_out.columns[int(Output)])
                    MODEL_SAVE_FOLDER_PATH = f'{MODEL_FOLDER_PATH}/{Folder}_LOW_{Low}'
                    # if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
                    #     os.mkdir(MODEL_SAVE_FOLDER_PATH)
            elif _ == 2:
                if Current == '':
                    pass
                else:
                    test_in.iloc[int(Current):, int(Input)] = test_in.iloc[int(Current), int(Input)]
                    test_out.iloc[int(Current):, int(Output)] = test_out.iloc[int(Current), int(Output)]
                    print(f'scenario: {file_}, CURRENT: {Current}', test_out.columns[int(Output)])
                    MODEL_SAVE_FOLDER_PATH = f'{MODEL_FOLDER_PATH}/{Folder}_CURRENT_{Current}'
                    # if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
                    #     os.mkdir(MODEL_SAVE_FOLDER_PATH)
            elif _ == 3:
                print(f'scenario: {file_}, Normal')
                MODEL_SAVE_FOLDER_PATH = f'{MODEL_FOLDER_PATH}/Normal'
                # if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
                #     os.mkdir(MODEL_SAVE_FOLDER_PATH)

            if _ == 0 or _ == 3 or (_ == 1 and Low != '') or (_ == 2 and Current != ''):
                test_in = pd.DataFrame(scaler_in.transform(test_in), columns=in_col)
                test_out = pd.DataFrame(scaler_out.transform(test_out), columns=out_col)
                predictions_test = vae.predict(test_in)
                p_test = pd.DataFrame(predictions_test, columns=out_col)

                p_test_scale = copy.deepcopy(p_test)

                p_test = pd.DataFrame(scaler_out.inverse_transform(p_test), columns=out_col)

                test_out_scale = copy.deepcopy(test_out)
                test_out = pd.DataFrame(scaler_out.inverse_transform(test_out), columns=out_col)

                mse_ = []
                for i, up_thresh in zip(range(output_dim), out_col):

                    mse = (test_out_scale.iloc[:, i] - p_test_scale.iloc[:, i]) ** 2
                    residual = pd.DataFrame({'Residual': mse})
                    print(len(test), len(residual))
                    a = temp_UCL[i]

                    coll = f'{MODEL_SAVE_FOLDER_PATH}/{ylabel[i]}'
                    if residual.iloc[len(test)-1:len(test), ].values[0][0] > a:
                        print(up_thresh)
                        over_threshold.append(up_thresh)
                        col_.append(coll)
                    mse_.append(residual)
                #                     break
                save_data.append([test_A_out, test_out, p_test])
                mse_cal.append(mse_)
            elif Low == '' or Current == '':
                save_data.append(['', '', '', ''])
                mse_cal.append([''])
        #             break
        save_data_one_scenario.append(save_data)
        mse_calcul.append(mse_cal)
    #         break
    save_data_all_scenario.append(save_data_one_scenario)
    mse_calculation.append(mse_calcul)
    # break
print("time :", time.time() - start)
result = pd.DataFrame([col_, over_threshold])
real_result = result.T
real_result.to_csv('result.csv')

dict_ = {}
dict_2 = {}
fault_type = ['high', 'low', 'crruent', 'normal']
for key_1, num_1 in zip(allFiles_name, range(len(allFiles_name))):
    dict_[f'{key_1}'] = {key : [] for key in dict.fromkeys(ylabel).keys()}
    dict_2[f'{key_1}'] = {key : [] for key in dict.fromkeys(ylabel).keys()}
    for key_2, num_2 in zip(ylabel, range(len(ylabel))):
        dict_[f'{key_1}'][f'{key_2}'] = {key : [] for key in dict.fromkeys(fault_type).keys()}
        dict_2[f'{key_1}'][f'{key_2}'] = {key : [] for key in dict.fromkeys(fault_type).keys()}
        for key_3, num_3 in zip(fault_type, range(len(fault_type))):
            dict_[f'{key_1}'][f'{key_2}'][f'{key_3}'] = save_data_all_scenario[num_1][num_2][num_3]
            dict_2[f'{key_1}'][f'{key_2}'][f'{key_3}'] = mse_calculation[num_1][num_2][num_3]

with open(save_db, 'wb') as f:
    pickle.dump(dict_, f)
with open(save_db2, 'wb') as f:
    pickle.dump(dict_2, f)