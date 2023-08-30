import itertools

import numpy as np
import torch
import torch.nn as nn


class NkELAnn(nn.Module):
    def __init__(self, para_train_data: np.ndarray = None,
                 para_train_target: np.ndarray = None,
                 para_test_data: np.ndarray = None,
                 para_parallel_layer_nodes=[],
                 para_loops: int = 300,
                 para_k_label: int = 3,
                 para_label_select=[],
                 para_label_num: int = 0,
                 para_learning_rate: float = 0.01,
                 para_activators: str = "s" * 100):
        """
        Create the neural network model.

        @param para_train_data: Training data feature matrix.
        @param para_train_target: Training class matrix.
        @param para_test_data: Testing data feature matrix.
        @param para_parallel_layer_nodes: Node list of hidden layers.
        @param para_loops: Training bounds.
        @param para_k_label: Value of k.
        @param para_label_select: Labelsets.
        @param para_label_num: Number of label.
        @param para_learning_rate: Learning rate.
        @param para_activators: String of activators.
        """
        super().__init__()
        self.device = torch.device("cuda")
        self.train_data = para_train_data
        self.train_target = para_train_target
        self.test_data = para_test_data
        self.learning_rate = para_learning_rate
        self.loops = para_loops
        self.k_labels = para_k_label
        self.label_select = para_label_select
        self.classifier_num = len(self.label_select)
        self.label_num = para_label_num

        # Parallel part
        self.parallel_model = [ParallelAnn(para_parallel_layer_nodes, para_activators).to(self.device)
                               for _ in range(self.classifier_num)]
        self.optimizer = torch.optim.Adam(itertools.chain(*[model.parameters() for model in self.parallel_model]),
                                          lr=para_learning_rate)
        self.loss_function = nn.MSELoss().to(self.device)
        self.soft_max = nn.Softmax(dim=1)
        pass

    def forward(self, para_input: np.ndarray = None):
        temp_input = torch.from_numpy(para_input).float().to(self.device)
        temp_parallel_output = [model(temp_input) for model in self.parallel_model]
        self.parallel_output = temp_parallel_output
        temp_output = temp_parallel_output[0]
        for i in range(len(temp_parallel_output) - 1):
            temp_output = torch.cat((temp_output, temp_parallel_output[i + 1]), 0)
        return temp_output

    def one_round_train(self, para_input: np.ndarray = None):
        """
        A round traning process.

        @param para_input: Training data feature matrix.
        @return: Loss value.
        """

        # Compute prediction error
        temp_out = self(para_input)
        temp_loss = self.loss_function(temp_out, self.train_target)

        # Backpropagation
        self.optimizer.zero_grad()
        temp_loss.backward()
        self.optimizer.step()

        return temp_loss.item()

    def fit(self):
        """
        The training process.
        """
        i_counts = 1
        temp_loops = self.loops

        while i_counts <= temp_loops:
            temp_loss_value = self.one_round_train(self.train_data)
            i_counts += 1
            if temp_loss_value <= 0.0002:
                print("temp_loss_value", temp_loss_value)
                print("i_counts", i_counts)
                break
            # last_loss = temp_loss_value
            if i_counts == temp_loops:
                print("final temp_loss_value", temp_loss_value)
        pass

    def predict(self):
        """
        Prediction.

        @return: the numerical prediction.
        """
        self(self.test_data)
        temp_predict_list = self.parallel_output
        temp_label_subsets_num = self.classifier_num
        temp_target_shape = (self.test_data.shape[0], self.label_num)
        result_matrix = np.zeros(temp_target_shape)
        temp_instance_num = self.test_data.shape[0]

        for i in range(temp_label_subsets_num):
            temp_predict_result = temp_predict_list[i]
            temp_label_subset = self.label_select[i]
            temp_predict_prob_result = self.soft_max(temp_predict_result).to(self.device).cpu().detach().numpy()
            temp_class_label_matrix = transform_class_label(self.k_labels)
            temp_predict = np.zeros((temp_instance_num, self.k_labels))
            temp_result_matrix = np.zeros(temp_target_shape)

            for j in range(self.k_labels):
                temp_index = temp_class_label_matrix[j]
                for k in temp_index:
                    temp_predict[:, j] += temp_predict_prob_result[:, k]
                temp_result_matrix[:, temp_label_subset[j]] = temp_predict[:, j]
                #   求各个网络的预测值的和
            result_matrix += temp_result_matrix
        return result_matrix

    pass


class ParallelAnn(nn.Module):
    def __init__(self, para_parallel_layer_nodes=[], para_activators: str = "s" * 100):
        """
         Construct a parallel structure neural network

        @param para_parallel_layer_nodes: The nodes of hidden layer
        @param para_activators: Activator function string
        """
        super().__init__()
        temp_model = []
        for i in range(len(para_parallel_layer_nodes) - 1):
            temp_input = para_parallel_layer_nodes[i]
            temp_out = para_parallel_layer_nodes[i + 1]
            temp_linear_model = nn.Linear(temp_input, temp_out)
            temp_model.append(temp_linear_model)
            temp_model.append(get_activator(para_activators[i]))
            pass
        self.model = nn.Sequential(*temp_model)

        pass

    def forward(self, para_input: torch.tensor = None):
        temp_output = self.model(para_input)

        return temp_output

    pass


def get_activator(para_activator: str = 's'):
    """
    Parsing the specific char of activator.

    @param para_activator: Char of activator
    @return: Activator function
    """
    if para_activator == 'r':
        return nn.ReLU()
    elif para_activator == 's':
        return nn.Sigmoid()
    elif para_activator == 't':
        return nn.Tanh()
    else:
        return nn.Sigmoid()


def transform_class_label(para_label_select_len: int = 0):
    """
    Transform class to labels.
    1 2 3     class
    0 0 0 --> 1
    1 0 0 --> 2
    0 1 0 --> 3
    1 1 0 --> 4
    0 0 1 --> 5
    1 0 1 --> 6
    0 1 1 --> 7
    1 1 1 --> 8
    So for the label 1 label we will get [2,4,6,8]
    Note: due to the characteristic of python, we start from class 0
    @param para_label_select_len: Value of k
    @return:A matrix
    """
    temp_matrix = []
    for i in range(2 ** para_label_select_len):
        temp_index_value = i
        tem_list = []
        for j in range(para_label_select_len):
            tem_list.append(temp_index_value % 2)
            temp_index_value //= 2
        temp_matrix.append(tem_list)
    temp_matrix = np.array(temp_matrix).transpose()

    result_matrix = []
    for i in range(para_label_select_len):
        temp_class_index = np.argwhere(temp_matrix[i] > 0).flatten().tolist()
        result_matrix.append(temp_class_index)
    return np.array(result_matrix)
