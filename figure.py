import pandas as pd
import pickle
import matplotlib.pyplot as plt
import glob
import os

save_db, save_db2 = 'Noise_0.01_3sigma/All_dict.pkl', 'Noise_0.01_3sigma/All_dict2.pkl'

para = pd.read_csv('new_corr_PARA.csv', na_filter=False)
in_col, out_col = list(para[(para['total'] == '0')]['CNS_']), list(para[(para['OUT'] == '1')]['CNS'])
ylabel = list(para[(para['OUT'] == '1')]['DESC'])

output_length = len(out_col)
current = para['CURRENT'][:output_length]
high, low = para['HIGH'][:output_length], para['LOW'][:output_length]
input_num, output_num = para['input'][:output_length], para['output'][:output_length]

with open(save_db, 'rb') as f:
    dict_ = pickle.load(f)

with open(save_db2, 'rb') as f:
    dict_2 = pickle.load(f)

temp_UCL = pd.read_csv('Noise_0.01_3sigma/threshold.csv')

allFiles = glob.glob("../../YH_LOCA_CSV/*.csv")    #   이거만 바꾸면됨

allFiles_name = []
for i in range(len(allFiles)):
    allFiles_name.append(allFiles[i][18:-4])

# select = 32
# allFiles_name = allFiles_name[select:select + 2]

'''
dict_[0][0][0][0] 여기에서
처음 [] 부분은 시나리오 명
두번째 [] 부분은 신호 데이터 명

세번째 [] 부분은 신호 데이터 fault에 대한 high low current normal 인데
low 또는 crrent가 없을수도있어.... 

마지막 [] 부분은 순서대로 실제, 고장시, 재구성 임
이거를 기준으로 그래프 그리면됨

dict_2[0][0][0][0] 여기는
dict_과 마지막만 다름
마지막 [] 부분은 데이터에서 칼럼에 해당됨 총 26개
'''
fault_type = ['high', 'low', 'crruent', 'normal']

MODEL_FOLDER = 'Noise_001_3sigmafig'
fig = plt.figure(constrained_layout=True, figsize=(4, 5))     # figure 그리기
gs = fig.add_gridspec(8, 1)                                   # figure 안에 subplot이 있어서
axs = [fig.add_subplot(gs[0:4, :]),  # 1                      # subplot을 clear하는 방법 적용
       fig.add_subplot(gs[4:8, :]),  # 2
       ]

if not os.path.exists(MODEL_FOLDER):
    os.mkdir(MODEL_FOLDER)
for key_1, num_1 in zip(allFiles_name, range(len(allFiles_name))):
    NORMAL_triger = True
    for key_2, num_2, Current, High, Low in zip(ylabel, range(len(ylabel)), current, high, low):
        for key_3, num_3 in zip(fault_type, range(len(fault_type))):
            LOW_triger = True
            CURRENT_triger = True

            MODEL_FOLDER_PATH = f'{MODEL_FOLDER}/{key_1}'
            if not os.path.exists(MODEL_FOLDER_PATH):
                os.mkdir(MODEL_FOLDER_PATH)
            # 고장 주입 high
            if num_3 == 0:
                print(f'{key_1[:]}', '===', f'{key_2}', '===', f'{key_3}')
                MODEL_SAVE_FOLDER_PATH = f'{MODEL_FOLDER_PATH}/{key_2}_HIGH_{High}'
                if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
                    os.mkdir(MODEL_SAVE_FOLDER_PATH)
            elif num_3 == 1:
                if Low == '':
                    LOW_triger = False
                    pass
                else:
                    print(f'{key_1[:]}', '===', f'{key_2}', '===', f'{key_3}')
                    MODEL_SAVE_FOLDER_PATH = f'{MODEL_FOLDER_PATH}/{key_2}_LOW_{Low}'
                    if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
                        os.mkdir(MODEL_SAVE_FOLDER_PATH)
            elif num_3 == 2:
                if Current == '':
                    CURRENT_triger = False
                    pass
                else:
                    print(f'{key_1[:]}', '===', f'{key_2}', '===', f'{key_3}')
                    MODEL_SAVE_FOLDER_PATH = f'{MODEL_FOLDER_PATH}/{key_2}_CURRENT_{Current}'
                    if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
                        os.mkdir(MODEL_SAVE_FOLDER_PATH)
            elif num_3 == 3:
                if NORMAL_triger:
                    NORMAL_triger = False
                    print(f'{key_1[:]}', '===', f'{key_2}', '===', f'{key_3}')
                    MODEL_SAVE_FOLDER_PATH = f'{MODEL_FOLDER_PATH}/Normal'
                    if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
                        os.mkdir(MODEL_SAVE_FOLDER_PATH)
                else:
                    CURRENT_triger = False

            if LOW_triger and CURRENT_triger:
                for i in range(len(ylabel)):
                    axs[0].cla()           #subplot clear
                    axs[1].cla()           #subplot clear

                    axs[0].plot(dict_[f'{key_1}'][f'{key_2}'][f'{key_3}'][0].iloc[:, i],
                                marker='o', markersize=4, color='g', label='Normal')
                    axs[0].plot(dict_[f'{key_1}'][f'{key_2}'][f'{key_3}'][1].iloc[:, i],
                                linewidth=0.5, color='b', label='Faulted')
                    axs[0].plot(dict_[f'{key_1}'][f'{key_2}'][f'{key_3}'][2].iloc[:, i],
                                marker='o', markersize=1, color='r', label='Reconstructed')
                    axs[0].legend(loc=1, fontsize=8)
                    axs[0].set_ylabel(ylabel[i])
                    axs[0].tick_params(labelsize=10)
                    axs[0].grid()
                    #
                    axs[1].plot(dict_2[f'{key_1}'][f'{key_2}'][f'{key_3}'][i])
                    axs[1].axhline(y=temp_UCL.iloc[:, 1][i], color='r')
                    # axs[1].axhline(y=b, color='r')
                    axs[1].set_ylabel('Reconstruction Error')
                    axs[1].set_xlabel('Time(s)')
                    axs[1].tick_params(labelsize=10)
                    axs[1].grid()

                    fig.savefig(fname=f'{MODEL_SAVE_FOLDER_PATH}/{ylabel[i]}.png', dpi=600, facecolor=None)
            #                     plt.show()

            #             break
            else:
                print('No data')