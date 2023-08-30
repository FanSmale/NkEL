
import json
import os

import numpy as np
import scipy.io as scio
import torch
from sklearn import preprocessing
from sklearn.model_selection import KFold

from NkEL import NkEL


class Demo:
    def __init__(self, para_file, para_output_file="", para_k_label=4, para_nodes=[], para_loops=300):
        """
        Construct a Demo

        @param para_file: Data file path.
        @param para_output_file: Print the measures
        @param para_k_label: Value of k.
        @param para_nodes: The nodes of hidden layer
        @param para_loops: The number of training bound.
        """
        assert scio.loadmat(para_file), print("读取文件失败")

        all_data = scio.loadmat(para_file)
        self.file_path = para_output_file
        self.loops = para_loops
        train_data = all_data['data']
        self.train_target = all_data['targets']
        min_max_scaler = preprocessing.MinMaxScaler()
        self.train_data = min_max_scaler.fit_transform(np.array(train_data))  # 归一化
        self.train_target[self.train_target < 0] = 0
        self.train_target_t = self.train_target.transpose()
        self.device = torch.device('cuda')
        self.k_label = para_k_label
        self.node_list = para_nodes

    def k_fold_test(self):
        kf = KFold(n_splits=5, shuffle=True)
        construct_time_list = []
        macro_list = []
        micro_list = []
        peak_f1_list = np.array([])
        ndcg_list = np.array([])
        auc_list = np.array([])
        temp_feature_num = self.train_data.shape[1]
        temp_node_list = [temp_feature_num]
        for i in self.node_list:
            temp_node_list.append(i)
        temp_node_list.append(2 ** self.k_label)
        for train_index, test_index in kf.split(self.train_data):
            temp_train_data = self.train_data[train_index, :]
            temp_test_data = self.train_data[test_index, :]
            temp_train_target = self.train_target[train_index, :]
            temp_test_target = self.train_target[test_index, :]

            nkel_ann = NkEL(temp_train_data, temp_train_target,
                           temp_test_data, temp_test_target,
                           self.k_label, temp_node_list,self.loops).to(self.device)
            # train
            nkel_ann.fit()
            # predict
            nkel_ann.predict()
            # nkel_ann.ensemble_test()

            peak_f1 = nkel_ann.compute_peak_f1()
            ndcg = nkel_ann.compute_ndcg()
            auc = nkel_ann.compute_auc()
            macro, micro = nkel_ann.compute_macro_micro_f1()

            # 计算评价指标值
            peak_f1_list = np.append(peak_f1_list, peak_f1)
            ndcg_list = np.append(ndcg_list, ndcg)
            auc_list = np.append(auc_list, auc)
            macro_list.append(macro)
            micro_list.append(micro)
            print("************************")

        macro_arr = np.array(macro_list)
        micro_arr = np.array(micro_list)

        mean_peak_f1 = np.mean(peak_f1_list)
        std_peak_f1 = np.std(peak_f1_list)
        mean_ndcg = np.mean(ndcg_list)
        std_ndcg = np.std(ndcg_list)
        mean_auc = np.mean(auc_list)
        std_auc = np.std(auc_list)
        mean_macrof1 = np.mean(macro_arr)
        std_macrof1 = np.std(macro_arr)
        mean_microf1 = np.mean(micro_arr)
        std_microf1 = np.std(micro_arr)

        print("peak_f1:", round(mean_peak_f1, 4), round(std_peak_f1, 4))
        print("ndcg:", round(mean_ndcg, 4), round(std_ndcg, 4))
        print("auc:", round(mean_auc, 4), round(std_auc, 4))
        print("macro-f1:", round(mean_macrof1, 4), round(std_macrof1, 4))
        print("micro-f1:", round(mean_microf1, 4), round(std_microf1, 4))

        with open(self.file_path + ".txt", "w") as f:
            f.write("peak_f1: " + str(round(mean_peak_f1, 4)) + "  " + str(round(std_peak_f1, 4)) + "\n")
            f.write("ndcg: " + str(round(mean_ndcg, 4)) + "  " + str(round(std_ndcg, 4)) + "\n")
            f.write("auc: " + str(round(mean_auc, 4)) + "  " + str(round(std_auc, 4)) + "\n")
            f.write("macro_f1: " + str(round(mean_macrof1, 4)) + "  " + str(round(std_macrof1, 4)) + "\n")
            f.write("micro_f1: " + str(round(mean_microf1, 4)) + "  " + str(round(std_microf1, 4)) + "\n")


if __name__ == '__main__':

    lists = ['Emotions']
    # check whether the JSON file exists
    config_name = '../configuration./config.json'
    assert os.path.exists(config_name), 'Config file is not accessible.'
    # open json
    with open(config_name) as f:
        cfg = json.load(f)['nkel']
    output_file_path = '../result/'
    for file_name in lists:
        print("begin: ***************************************************", file_name)
        file_path = cfg[file_name]['fileName']
        k_labels = cfg[file_name]['kLabel']
        hidden_layer_nodes = cfg[file_name]['hiddenLayerNumNodes']
        loops = cfg[file_name]['loops']
        nkel = Demo(file_path, output_file_path + file_name,
                         k_labels, hidden_layer_nodes, loops)
        nkel.k_fold_test()
        print("end: ***************************************************", file_name)
