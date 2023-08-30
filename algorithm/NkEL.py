import random

import math
import numpy as np
import torch
import torch.nn as nn

from NkELAnn import NkELAnn


class NkEL(nn.Module):
    def __init__(self, para_train_data: np.ndarray = None,
                 para_train_target: np.ndarray = None,
                 para_test_data: np.ndarray = None,
                 para_test_target: np.ndarray = None,
                 para_k_labels: int = 4,
                 para_parallel_layer_nodes=[],
                 para_loops: int = 300,
                 para_learning_rate: float = 0.01,
                 para_activators: str = "s" * 100):
        """
        Create the neural network model.

        @param para_train_data: Training data feature matrix.
        @param para_train_target: Training label matrix.
        @param para_test_data: Testing data feature matrix.
        @param para_test_target: Testing label matrix.
        @param para_k_labels: Value of k.
        @param para_parallel_layer_nodes: Node list of hidden layer.
        @param para_learning_rate: Learning rate.
        @param para_activators: Activator string
        """
        super().__init__()
        self.device = torch.device("cuda")
        self.train_data = para_train_data
        self.train_target = para_train_target
        self.test_data = para_test_data
        self.test_target = para_test_target
        self.learning_rate = para_learning_rate
        self.train_target_t = self.train_target.T
        self.k_labels = para_k_labels
        self.label_select = []
        self.label_num = np.size(self.train_target, 1)
        self.threshold_value = 0.5
        self.predict_prob_matrix = None
        self.predict_label_matrix = None
        self.macro_f1 = 0
        self.micro_f1 = 0
        self.f1_score = 0
        self.train_target_loss = None
        self.parallel_output = []
        self.label_embedding_num = np.zeros(self.train_target.shape[1])
        self.construct_time = 0

        self.get_nearest_label_subset()  # k-random strategy
        self.nkel_ann = NkELAnn(self.train_data, self.train_target_loss,
                                    self.test_data, para_parallel_layer_nodes,
                                    para_loops, self.k_labels,
                                    self.label_select, self.label_num,
                                    para_learning_rate, para_activators
                                    ).to(self.device)
        pass

    def fit(self):
        """
        The training process.
        """
        self.nkel_ann.fit()
        pass

    def predict(self):
        """
        Get the probability of each label.
        """
        result_matrix = self.nkel_ann.predict()
        # 求多个网络的预测均值
        for i in range(self.train_target.shape[1]):
            result_matrix[:, i] = result_matrix[:, i] / self.label_embedding_num[i]
        self.predict_prob_matrix = result_matrix
        predict_label_matrix = np.array(result_matrix >= self.threshold_value, dtype=int)
        self.predict_label_matrix = predict_label_matrix
        pass

    def get_euclidean_dis(self):
        """
        Get the euclidean distance matrix between labels.

        @return: Distance matrix.
        """
        sum_x = np.sum(np.square(self.train_target_t), 1)
        dis = np.add(np.add(-2 * np.dot(self.train_target_t, self.train_target_t.T), sum_x).T, sum_x)
        return np.sqrt(dis)

    def get_nearest_label_subset(self):
        """
        Construct the labelsets according to the distance.
        """
        print("select the nearest")
        distance_matrix = self.get_euclidean_dis()
        # distance_matrix = self.get_jaccard_dis()
        temp_label = np.size(distance_matrix, 0)
        result = []
        temp_select_list = []
        temp_train_target_list = []
        temp_label_index_length = len(str(temp_label))
        for i in range(temp_label):
            temp_distance_index = (np.argsort(distance_matrix[i])).tolist()
            temp_list = [i]
            temp_distance_index.remove(i)
            temp_index = 0
            while len(temp_list) < self.k_labels:
                index_ = temp_distance_index[temp_index]
                if index_ not in temp_list:
                    temp_list.append(index_)
                temp_index += 1
            temp_list = (-1 * np.sort(-1 * np.array(temp_list))).tolist()
            temp_str = ''.join("0" * (temp_label_index_length - len(str(_))) + str(_) for _ in temp_list)
            # print(temp_list, "temp_sum", temp_str)
            if temp_str not in temp_select_list:
                temp_select_list.append(temp_str)
                result.append(temp_list)
                self.label_embedding_num[temp_list] += 1
                temp_train_target = self.transform_label_class(temp_list)
                temp_train_target_list.append(temp_train_target)
        self.label_select = np.array(result)
        temp_train_target_array = np.array(temp_train_target_list[0])
        if len(np.argwhere(np.array(self.label_embedding_num) == 0)) > 0:
            print("Some wrong with ", np.argwhere(np.array(self.label_embedding_num) == 0))
        for i in range(len(temp_train_target_list) - 1):
            temp_train_target_array = np.append(temp_train_target_array, temp_train_target_list[i + 1], axis=0)
            pass
        self.train_target_loss = torch.from_numpy(temp_train_target_array).float().to(self.device)
        pass

    def transform_label_class(self, para_list=[]):
        """
        Transform label value to class

        @param para_list: The labelset
        @return: The binary class value
        """
        temp_select = self.train_target[:, para_list]
        temp_instance_num, temp_label_num = temp_select.shape
        result_matrix = np.zeros((temp_instance_num, 2 ** temp_label_num), dtype=int)
        for i in range(temp_instance_num):
            temp_class = 0
            for j in range(temp_label_num):
                temp_class += temp_select[i][j] * (2 ** j)
            result_matrix[i][temp_class] = 1
            pass
        return result_matrix.tolist()

    def compute_peak_f1(self):
        """
        Get the peak_f1
        @return: peak_f1 value
        """
        temp_predict_vector = self.predict_prob_matrix.reshape(-1)
        temp_predict_sort_index = np.argsort(-temp_predict_vector)
        temp_test_target_vector = self.test_target.reshape(-1)
        temp_test_target_sort = temp_test_target_vector[temp_predict_sort_index]
        temp_f1_list = []
        TP_FN = np.sum(self.test_target > 0)
        for i in range(temp_predict_sort_index.size):
            TP = np.sum(temp_test_target_sort[0:i + 1] == 1)
            P = TP / (i + 1)
            R = TP / TP_FN
            temp_f1 = 0
            if (P + R) != 0:
                temp_f1 = 2.0 * P * R / (P + R)
                pass
            temp_f1_list.append(temp_f1)
            pass

        temp_f1_list = np.array(temp_f1_list)
        temp_max_f1_index = np.argmax(temp_f1_list)
        peak_f1 = np.max(temp_f1_list)
        threshold_value = temp_predict_vector[temp_max_f1_index]
        self.threshold_value = threshold_value
        print("compute_peak_f1:", peak_f1)
        return peak_f1
        pass

    def compute_auc(self):
        """
        Get the micro_auc
        @return: micro_auc value
        """

        temp_predict_vector = self.predict_prob_matrix.reshape(-1)
        temp_test_target_vector = self.test_target.reshape(-1)
        temp_predict_sort_index = np.argsort(temp_predict_vector)

        M, N = 0, 0
        for i in range(temp_predict_vector.size):
            if temp_test_target_vector[i] == 1:
                M += 1
            else:
                N = N + 1
                pass
            pass

        sigma = 0
        for i in range(temp_predict_vector.size - 1, -1, -1):
            if temp_test_target_vector[temp_predict_sort_index[i]] == 1:
                sigma += i + 1
                pass
            pass
        auc = (sigma - (M + 1) * M / 2) / (M * N)
        print("compute_auc:", auc)
        return auc

    def compute_ndcg(self):
        """
        Get the ndcg

        @return: ndcg value
        """
        temp_predict_vector = self.predict_prob_matrix.reshape(-1)
        temp_test_target_vector = self.test_target.reshape(-1)

        temp_predict_sort_index = np.argsort(-temp_predict_vector)
        temp_predict_target_sort = temp_test_target_vector[temp_predict_sort_index]

        temp_target_sort = np.sort(temp_test_target_vector)
        temp_target_sort = np.flipud(temp_target_sort)

        dcg = 0;
        for i in range(temp_predict_vector.size):
            rel = temp_predict_target_sort[i]
            denominator = math.log2(i + 2)
            dcg += rel / denominator

        idcg = 0
        for i in range(temp_predict_vector.size):
            rel = temp_target_sort[i]
            denominator = math.log2(i + 2)
            idcg += rel / denominator
        ndcg = dcg / idcg
        print("compute_ndcg: ", ndcg)
        return ndcg

    def compute_macro_micro_f1(self, para_threshold=0.5):
        """
        Get the macro_f1 and micro_f1

        @param para_threshold: threshold value
        @return:  macro_f1 and micro_f1
        """

        temp_predict = self.predict_prob_matrix.T
        temp_target = self.test_target.T
        temp_instance, temp_labels = self.predict_prob_matrix.shape
        temp_predict_label = np.array(temp_predict >= para_threshold, dtype=int)
        temp_sum_precision = 0
        temp_sum_recall = 0
        temp_sum_target_p = 0
        temp_sum_predict_p = 0
        temp_sum_tp = 0
        for i in range(temp_labels):
            temp_target_p = np.sum(temp_target[i] == 1)
            temp_predict_p = np.sum(temp_predict_label[i] == 1)
            temp_index = np.argwhere(temp_target[i] == 1).flatten()
            temp_tp = np.sum(temp_predict_label[i, temp_index] == 1)
            temp_sum_target_p += temp_target_p
            temp_sum_predict_p += temp_predict_p
            temp_sum_tp += temp_tp
            temp_precision, temp_recall = 0, 0
            if temp_predict_p != 0:
                temp_precision = temp_tp / temp_predict_p
            if temp_target_p != 0:
                temp_recall = temp_tp / temp_target_p
            temp_sum_precision += temp_precision
            temp_sum_recall += temp_recall
        macro_precision = temp_sum_precision / temp_labels
        macro_recall = temp_sum_recall / temp_labels
        if macro_precision + macro_recall != 0:
            self.macro_f1 = (2 * macro_precision * macro_recall) / (macro_precision + macro_recall)

        micro_precision, micro_recall = 0, 0
        if temp_sum_predict_p != 0:
            micro_precision = temp_sum_tp / temp_sum_predict_p

        if temp_sum_target_p != 0:
            micro_recall = temp_sum_tp / temp_sum_target_p

        if micro_precision + micro_recall != 0:
            self.micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall)

        print("macro_f1:", self.macro_f1)
        print("micro_f1:", self.micro_f1)
        return self.macro_f1, self.micro_f1

    pass
