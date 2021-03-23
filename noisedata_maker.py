import pandas as pd
import numpy as np
import pickle
import glob
from sklearn.preprocessing import MinMaxScaler


def creat_noise(data):
    creat_noise_data = np.random.normal(1, 0.01, data.shape)
    noise_data = data * creat_noise_data
    return noise_data


# columns
para = pd.read_csv('new_corr_PARA.csv')

in_col = list(para[(para['total'] == 0)]['CNS_'])
out_col = list(para[(para['OUT'] == 1)]['CNS'])
in_scale = 'scaler_in.pkl'
out_scale = 'scaler_out.pkl'
in_data_pkl = 'two_d_in_data.pkl'
out_data_pkl = 'two_d_out_data.pkl'

# True = 스케일 후 노이즈 / False = 스케일 전 노이즈
noise_after_scale = False

# train data
path = '../../YH_LOCA_CSV' # use your path
allFiles = glob.glob(path + "/*.csv")
Train_data = pd.DataFrame()
in_ = []
out_ = []

for file_ in allFiles:
    in_da = pd.read_csv(file_,index_col=None, header=0)
    in_da = in_da[in_col]
    out_da = in_da[out_col]
    in_.append(in_da)
    out_.append(out_da)
    print(file_)
in_data = pd.Series(in_)
out_data = pd.Series(out_)
print(np.shape(in_data))

if noise_after_scale:
    # max
    list_in = []
    for _ in range(len(in_data)):
        test_in = in_data[_].describe()
        test_in_max = test_in.iloc[7:,:]
        test_in_max_ = test_in_max.to_numpy()[0]
        list_in.append(test_in_max_)
    in_max = pd.DataFrame(list_in)
    max_in = in_max.describe().iloc[7:,:]  # max 중 max 찾기
    max_in.columns = in_data[0].columns

    # min
    list_in = []
    for _ in range(len(in_data)):
        test_in = in_data[_].describe()
        test_in_min = test_in.iloc[3:4,:]
        test_in_min_ = test_in_min.to_numpy()[0]
        list_in.append(test_in_min_)
    in_min = pd.DataFrame(list_in)
    min_in = in_min.describe().iloc[3:4,:]    # min 중 min 찾기
    min_in.columns = in_data[0].columns

    min_max_in = pd.concat([min_in, max_in])  # min_max_in
    min_max_out = min_max_in[out_col]   # min_max_out

    scaler_in = MinMaxScaler()
    scaler_in.fit(min_max_in)
    scaler_out = MinMaxScaler()
    scaler_out.fit(min_max_out)
    for _ in range(len(in_data)):
        in_data[_] = pd.DataFrame(scaler_in.transform(in_data[_]), columns=in_col)
        out_data[_] = pd.DataFrame(scaler_out.transform(out_data[_]), columns=out_col)
        print('{}.done'.format(_))

    with open(in_scale, 'wb') as f:
        pickle.dump(scaler_in, f)
    with open(out_scale, 'wb') as f:
        pickle.dump(scaler_out, f)

    noise_in_da = []
    noise_out_da = []
    for _ in range(len(in_data)):
        noise_in = in_data[_].apply(creat_noise)
        noise_out = noise_in[out_col]
        print(_ + 1, noise_in.shape, noise_out.shape, type(noise_in))
        noise_in_da.append(noise_in)
        noise_out_da.append(noise_out)

    noise_in_data = pd.Series(noise_in_da)
    noise_out_data = pd.Series(noise_out_da)

    print(np.shape(noise_in_data[0]))

    time_leg = 10
    two_d_in = []
    two_d_out = []

    for se in range(len(noise_in_data)):
        one_in = noise_in_data[se]
        one_out = noise_out_data[se]

        for _ in range(0, len(one_in) - time_leg):
            two_d_in.append(one_in.iloc[_:_ + 1, :].to_numpy()[0])
            two_d_out.append(one_out.iloc[_:_ + 1, :].to_numpy()[0])
        print(np.shape(two_d_in), np.shape(two_d_out), f'=={se}.done==')

    with open(in_data_pkl, 'wb') as f:
        pickle.dump(two_d_in, f)

    with open(out_data_pkl, 'wb') as f:
        pickle.dump(two_d_out, f)
else:
    noise_in_da = []
    noise_out_da = []
    for _ in range(len(in_data)):
        noise_in = in_data[_].apply(creat_noise)
        noise_out = noise_in[out_col]
        print(_ + 1, noise_in.shape, noise_out.shape, type(noise_in))
        noise_in_da.append(noise_in)
        noise_out_da.append(noise_out)

    noise_in_data = pd.Series(noise_in_da)
    noise_out_data = pd.Series(noise_out_da)

    print(np.shape(noise_in_data[0]))

    # max
    list_in = []
    for _ in range(len(noise_in_data)):
        test_in = noise_in_data[_].describe()
        test_in_max = test_in.iloc[7:,:]
        test_in_max_ = test_in_max.to_numpy()[0]
        list_in.append(test_in_max_)
    in_max = pd.DataFrame(list_in)
    max_in = in_max.describe().iloc[7:,:]  # max 중 max 찾기
    max_in.columns = in_data[0].columns

    # min
    list_in = []
    for _ in range(len(noise_in_data)):
        test_in = noise_in_data[_].describe()
        test_in_min = test_in.iloc[3:4,:]
        test_in_min_ = test_in_min.to_numpy()[0]
        list_in.append(test_in_min_)
    in_min = pd.DataFrame(list_in)
    min_in = in_min.describe().iloc[3:4,:]    # min 중 min 찾기
    min_in.columns = in_data[0].columns

    min_max_in = pd.concat([min_in, max_in])  # min_max_in
    min_max_out = min_max_in[out_col]   # min_max_out

    scaler_in = MinMaxScaler()
    scaler_in.fit(min_max_in)
    scaler_out = MinMaxScaler()
    scaler_out.fit(min_max_out)
    for _ in range(len(noise_in_data)):
        noise_in_data[_] = pd.DataFrame(scaler_in.transform(noise_in_data[_]), columns=in_col)
        noise_out_data[_] = pd.DataFrame(scaler_out.transform(noise_out_data[_]), columns=out_col)
        print('{}.done'.format(_))

    with open(in_scale, 'wb') as f:
        pickle.dump(scaler_in, f)
    with open(out_scale, 'wb') as f:
        pickle.dump(scaler_out, f)

    time_leg = 10
    two_d_in = []
    two_d_out = []

    for se in range(len(noise_in_data)):
        one_in = noise_in_data[se]
        one_out = noise_out_data[se]

        for _ in range(0, len(one_in) - time_leg):
            two_d_in.append(one_in.iloc[_:_ + 1, :].to_numpy()[0])
            two_d_out.append(one_out.iloc[_:_ + 1, :].to_numpy()[0])
        print(np.shape(two_d_in), np.shape(two_d_out), f'=={se}.done==')

    with open(in_data_pkl, 'wb') as f:
        pickle.dump(two_d_in, f)

    with open(out_data_pkl, 'wb') as f:
        pickle.dump(two_d_out, f)