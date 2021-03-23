import pandas as pd
import numpy as np
import pickle
import glob
import time

with open('Noise_0.01_3sigma/All_dict2.pkl', 'rb') as f:       # 파일 변경 All_dict2 만 사용됨
    dict_ = pickle.load(f)

temp_UCL = pd.read_csv('Noise_0.01_3sigma/threshold.csv')              # 변경

allFiles = glob.glob("../../YH_LOCA_CSV/*.csv")

allFiles_name = []
for i in range(len(allFiles)):
    allFiles_name.append(allFiles[i][18:-4])    #  allFiles의 길이 조절을


para = pd.read_csv('new_corr_PARA.csv', na_filter=False)
in_col, out_col = list(para[(para['total'] == '0')]['CNS_']), list(para[(para['OUT'] == '1')]['CNS'])
ylabel = list(para[(para['OUT'] == '1')]['DESC'])
output_length = len(out_col)
current = para['CURRENT'][:output_length]
high, low = para['HIGH'][:output_length], para['LOW'][:output_length]
input_num, output_num = para['input'][:output_length], para['output'][:output_length]

'''
dict_['12_10010_30_5']['FEEDWATER PUMP OUTLET PRESS']['high'][0]
['12_10010_30_5']     ['FEEDWATER PUMP OUTLET PRESS']               ['high']                         [0]
  시나리오 명                출력 변수 DESC          고장 타입(high, low, current, normal)       출력 변수(총 26개)
'''

start = time.time()

fault_type = ['high', 'low', 'crruent', 'normal']
save_list = []
for key_1, num_1 in zip(allFiles_name, range(len(allFiles_name))):
    for key_2, num_2, Current, High, Low in zip(ylabel, range(len(ylabel)), current, high, low):
        for key_3, num_3 in zip(fault_type, range(len(fault_type))):
            residual = dict_[key_1][key_2][key_3]
            df = pd.DataFrame()

            if len(residual[0]) > 10:
                st = key_1 + '/' + key_2 + '_' + key_3
                for i, up_thresh in zip(range(len(dict_[key_1][key_2][key_3])), out_col):
                    a = temp_UCL.iloc[:, 1][i]
                    if residual[i].iloc[len(residual[i]) - 1:len(residual[i]), ].values[0][0] > a and key_3 != 'normal':
                        if num_2 == i:
                            fault_fault = 1
                            type_1_error = ''
                            type_2_error = ''
                            normal = ''
                            save_list.append([up_thresh, fault_fault, type_1_error, type_2_error, normal, st])
                        else:
                            fault_fault = ''
                            type_1_error = ''
                            type_2_error = 1
                            normal = ''
                            save_list.append([up_thresh, fault_fault, type_1_error, type_2_error, normal, st])
                    else:
                        if num_2 == i and key_3 != 'normal':
                            fault_fault = ''
                            type_1_error = 1
                            type_2_error = ''
                            normal = ''
                            save_list.append([up_thresh, fault_fault, type_1_error, type_2_error, normal, st])
#                         else:
#                             fault_fault = ''
#                             normal = 1
#                             fault_normal = ''
#                             save_list.append([up_thresh, fault_fault, fault_normal, normal, st])

print("time :", time.time() - start)

test1 = pd.DataFrame(save_list, columns=['signal', 'fault-fault', 'type_1_error', 'type_2_error', 'normal', 'where'])
test1.to_csv('FIG_Noise_0.01_3sigma.csv')