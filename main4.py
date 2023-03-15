#!/usr/bin/env python
# coding: utf-8

# In[101]:


from numpy.random import seed
import csv
import sqlite3
import time
import numpy as np
import random
import pandas as pd
from pandas import DataFrame
import scipy.sparse as sp
import math
import copy

from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import KernelPCA

import sys
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pytorchtools import EarlyStopping
from pytorchtools import BalancedDataParallel
from radam import RAdam
import torch.nn.functional as F

import networkx as nx

import warnings

warnings.filterwarnings("ignore")

import os
# from tensorboardX import SummaryWriter

# In[102]:


seed = 0
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True


# In[103]:


def prepare(df_drug, feature_list, mechanism, action, drugA, drugB):
    d_label = {}
    d_feature = {}

    # Transfrom the interaction event to number
    d_event = [] # 把药物关系和作用升降拼接字符串存到d_event数组里面
    for i in range(len(mechanism)):
        d_event.append(mechanism[i] + " " + action[i])
#统计关系表d_event里面元素和他的频率
    count = {}
    for i in d_event:
        if i in count:
            count[i] += 1
        else:
            count[i] = 1#===========一共65个所以就是65分类问题.
    event_num = len(count)
    list1 = sorted(count.items(), key=lambda x: x[1], reverse=True)
    for i in range(len(list1)):
        d_label[list1[i][0]] = i # d_label记录药品的名字和他的索引对应.

    vector = np.zeros(       (len(np.array(df_drug['name']).tolist()), 0), dtype=float)  # vector=[]
    for i in feature_list:
        # vector = np.hstack((vector, feature_vector(i, df_drug, vector_size)))#1258*1258
        tempvec = feature_vector(i, df_drug) #搭建特征向量.传入特征的名字和药物df
        vector = np.hstack((vector, tempvec))#=========这个地方特征向量可以加入nlp里面embedding,可以 不止用jaccard.*******************************************************************************或者单纯one-hot编码.等方式都可以.#=====================tempvec可以修改!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Transfrom the drug ID to feature vector
    for i in range(len(np.array(df_drug['name']).tolist())):
        d_feature[np.array(df_drug['name']).tolist()[i]] = vector[i]

    # Use the dictionary to obtain feature vector and label
    new_feature = []
    new_label = []

    for i in range(len(d_event)):
        temp = np.hstack((d_feature[drugA[i]], d_feature[drugB[i]]))
        new_feature.append(temp)
        new_label.append(d_label[d_event[i]])

    new_feature = np.array(new_feature)  # 323539*....
    new_label = np.array(new_label)  # 323539

    return new_feature, new_label, event_num

# In[104]:


def feature_vector(feature_name, df):
    def Jaccard(matrix):
        matrix = np.mat(matrix)

        numerator = matrix * matrix.T

        denominator = np.ones(np.shape(matrix)) * matrix.T + matrix * np.ones(np.shape(matrix.T)) - matrix * matrix.T

        return numerator / denominator

    all_feature = [] # 记录所有的特征
    drug_list = np.array(df[feature_name]).tolist()
    # Features for each drug, for example, when feature_name is target, drug_list=["P30556|P05412","P28223|P46098|……"]
    for i in drug_list:
        for each_feature in i.split('|'):
            if each_feature not in all_feature:
                all_feature.append(each_feature)  # obtain all the features
    feature_matrix = np.zeros((len(drug_list), len(all_feature)), dtype=float)
    df_feature = DataFrame(feature_matrix, columns=all_feature)  # Consrtuct feature matrices with key of dataframe
    for i in range(len(drug_list)):
        for each_feature in df[feature_name].iloc[i].split('|'): #对应特征位置写入1. # 这地方应该用批量写入能优化速度.
            df_feature[each_feature].iloc[i] = 1

    df_feature = np.array(df_feature)  #=================这地方加入特征
    sim_matrix = np.array(Jaccard(df_feature))

    print(feature_name + " len is:" + str(len(sim_matrix[0])))
    return np.hstack([sim_matrix,df_feature])


# In[105]:


class DDIDataset(Dataset):
    def __init__(self, x, y):
        self.len = len(x)
        self.x_data = torch.from_numpy(x)

        self.y_data = torch.from_numpy(y)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


# In[106]:


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, input_dim, n_heads, ouput_dim=None):

        super(MultiHeadAttention, self).__init__()
        self.d_k = self.d_v = input_dim // n_heads
        self.n_heads = n_heads
        if ouput_dim == None:
            self.ouput_dim = input_dim
        else:
            self.ouput_dim = ouput_dim
        self.W_Q = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_K = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_V = torch.nn.Linear(input_dim, self.d_v * self.n_heads, bias=False)
        self.fc = torch.nn.Linear(self.n_heads * self.d_v, self.ouput_dim, bias=False)

    def forward(self, X):
        ## (S, D) -proj-> (S, D_new) -split-> (S, H, W) -trans-> (H, S, W)
        Q = self.W_Q(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        K = self.W_K(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        V = self.W_V(X).view(-1, self.n_heads, self.d_v).transpose(0, 1)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        # context: [n_heads, len_q, d_v], attn: [n_heads, len_q, len_k]
        attn = torch.nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        # context: [len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(-1, self.n_heads * self.d_v)
        output = self.fc(context)
        return output


# In[107]:


class EncoderLayer(torch.nn.Module):
    def __init__(self, input_dim, n_heads):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(input_dim, n_heads)
        self.AN1 = torch.nn.LayerNorm(input_dim)

        self.l1 = torch.nn.Linear(input_dim, input_dim)
        self.AN2 = torch.nn.LayerNorm(input_dim)

    def forward(self, X):
        output = self.attn(X)
        X = self.AN1(output + X)

        output = self.l1(X)
        X = self.AN2(output + X)

        return X


# In[108]:


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


# In[109]:


class AE1(torch.nn.Module):  # Joining together
    def __init__(self, vector_size):
        super(AE1, self).__init__()

        self.vector_size = vector_size

        self.l1 = torch.nn.Linear(self.vector_size, (self.vector_size + len_after_AE) // 2)
        self.bn1 = torch.nn.BatchNorm1d((self.vector_size + len_after_AE) // 2)

        self.att2 = EncoderLayer((self.vector_size + len_after_AE) // 2, bert_n_heads)
        self.l2 = torch.nn.Linear((self.vector_size + len_after_AE) // 2, len_after_AE)

        self.l3 = torch.nn.Linear(len_after_AE, (self.vector_size + len_after_AE) // 2)
        self.bn3 = torch.nn.BatchNorm1d((self.vector_size + len_after_AE) // 2)

        self.l4 = torch.nn.Linear((self.vector_size + len_after_AE) // 2, self.vector_size)

        self.dr = torch.nn.Dropout(drop_out_rating)
        self.ac = gelu

    def forward(self, X):
        X = self.dr(self.bn1(self.ac(self.l1(X))))

        X = self.att2(X)
        X = self.l2(X)

        X_AE = self.dr(self.bn3(self.ac(self.l3(X))))

        X_AE = self.l4(X_AE)

        return X, X_AE


# In[110]:


class AE2(torch.nn.Module):  # twin network
    def __init__(self, vector_size):
        super(AE2, self).__init__()

        self.vector_size = vector_size // 2

        self.l1 = torch.nn.Linear(self.vector_size, (self.vector_size + len_after_AE // 2) // 2)
        self.bn1 = torch.nn.BatchNorm1d((self.vector_size + len_after_AE // 2) // 2)

        self.att2 = EncoderLayer((self.vector_size + len_after_AE // 2) // 2, bert_n_heads)
        self.l2 = torch.nn.Linear((self.vector_size + len_after_AE // 2) // 2, len_after_AE // 2)

        self.l3 = torch.nn.Linear(len_after_AE // 2, (self.vector_size + len_after_AE // 2) // 2)
        self.bn3 = torch.nn.BatchNorm1d((self.vector_size + len_after_AE // 2) // 2)

        self.l4 = torch.nn.Linear((self.vector_size + len_after_AE // 2) // 2, self.vector_size)

        self.dr = torch.nn.Dropout(drop_out_rating)

        self.ac = gelu

    def forward(self, X):
        X1 = X[:, 0:self.vector_size]
        X2 = X[:, self.vector_size:]

        X1 = self.dr(self.bn1(self.ac(self.l1(X1))))
        X1 = self.att2(X1)
        X1 = self.l2(X1)
        X_AE1 = self.dr(self.bn3(self.ac(self.l3(X1))))
        X_AE1 = self.l4(X_AE1)

        X2 = self.dr(self.bn1(self.ac(self.l1(X2))))
        X2 = self.att2(X2)
        X2 = self.l2(X2)
        X_AE2 = self.dr(self.bn3(self.ac(self.l3(X2))))
        X_AE2 = self.l4(X_AE2)

        X = torch.cat((X1, X2), 1)
        X_AE = torch.cat((X_AE1, X_AE2), 1)

        return X, X_AE


# In[111]:


class cov(torch.nn.Module):
    def __init__(self, vector_size):
        super(cov, self).__init__()

        self.vector_size = vector_size

        self.co2_1 = torch.nn.Conv2d(1, 1, kernel_size=(2, cov2KerSize))
        self.co1_1 = torch.nn.Conv1d(1, 1, kernel_size=cov1KerSize)
        self.pool1 = torch.nn.AdaptiveAvgPool1d(len_after_AE)

        self.ac = gelu

    def forward(self, X):
        X1 = X[:, 0:self.vector_size // 2]
        X2 = X[:, self.vector_size // 2:]

        X = torch.cat((X1, X2), 0)

        X = X.view(-1, 1, 2, self.vector_size // 2)

        X = self.ac(self.co2_1(X))

        X = X.view(-1, self.vector_size // 2 - cov2KerSize + 1, 1)
        X = X.permute(0, 2, 1)
        X = self.ac(self.co1_1(X))

        X = self.pool1(X)

        X = X.contiguous().view(-1, len_after_AE)

        return X


# In[112]:


class ADDAE(torch.nn.Module):
    def __init__(self, vector_size):
        super(ADDAE, self).__init__()

        self.vector_size = vector_size // 2

        self.l1 = torch.nn.Linear(self.vector_size, (self.vector_size + len_after_AE) // 2)
        self.bn1 = torch.nn.BatchNorm1d((self.vector_size + len_after_AE) // 2)

        self.att1 = EncoderLayer((self.vector_size + len_after_AE) // 2, bert_n_heads)
        self.l2 = torch.nn.Linear((self.vector_size + len_after_AE) // 2, len_after_AE)
        # self.att2=EncoderLayer(len_after_AE//2,bert_n_heads)

        self.l3 = torch.nn.Linear(len_after_AE, (self.vector_size + len_after_AE) // 2)
        self.bn3 = torch.nn.BatchNorm1d((self.vector_size + len_after_AE) // 2)

        self.l4 = torch.nn.Linear((self.vector_size + len_after_AE) // 2, self.vector_size)

        self.dr = torch.nn.Dropout(drop_out_rating)

        self.ac = gelu

    def forward(self, X):
        X1 = X[:, 0:self.vector_size]
        X2 = X[:, self.vector_size:]
        X = X1 + X2

        X = self.dr(self.bn1(self.ac(self.l1(X))))

        X = self.att1(X)
        X = self.l2(X)

        X_AE = self.dr(self.bn3(self.ac(self.l3(X))))

        X_AE = self.l4(X_AE)
        X_AE = torch.cat((X_AE, X_AE), 1)

        return X, X_AE


# In[113]:


class BERT(torch.nn.Module):
    def __init__(self, input_dim, n_heads, n_layers, event_num):# 第一个参数是 特征维度, 最后一个是输出维度.
        super(BERT, self).__init__()

        self.ae1 = AE1(input_dim)  # Joining together 编码
        self.ae2 = AE2(input_dim)  # twin loss
        self.cov = cov(input_dim)  # cov
        self.ADDAE = ADDAE(input_dim)

        self.dr = torch.nn.Dropout(drop_out_rating)
        self.input_dim = input_dim

        self.layers = torch.nn.ModuleList([EncoderLayer(len_after_AE * 5, n_heads) for _ in range(n_layers)])
        self.AN = torch.nn.LayerNorm(len_after_AE * 5)

        self.l1 = torch.nn.Linear(len_after_AE * 5, (len_after_AE * 5 + event_num) // 2)
        self.bn1 = torch.nn.BatchNorm1d((len_after_AE * 5 + event_num) // 2)

        self.l2 = torch.nn.Linear((len_after_AE * 5 + event_num) // 2, event_num)

        self.ac = gelu

    def forward(self, X):
        X1, X_AE1 = self.ae1(X)
        X2, X_AE2 = self.ae2(X)

        X3 = self.cov(X)

        X4, X_AE4 = self.ADDAE(X)

        X5 = X1 + X2 + X3 + X4

        X = torch.cat((X1, X2, X3, X4, X5), 1)

        for layer in self.layers:
            X = layer(X)
        X = self.AN(X)

        X = self.dr(self.bn1(self.ac(self.l1(X))))

        X = self.l2(X)

        return X, X_AE1, X_AE2, X_AE4


# In[114]:


class focal_loss(nn.Module):# 对交叉熵的优化. exp光滑处理.
    def __init__(self, gamma=2):
        super(focal_loss, self).__init__()

        self.gamma = gamma

    def forward(self, preds, labels):
        # assert preds.dim() == 2 and labels.dim()==1
        labels = labels.view(-1, 1)  # [B * S, 1]
        preds = preds.view(-1, preds.size(-1))  # [B * S, C]

        preds_logsoft = F.log_softmax(preds, dim=1)  # 先softmax, 然后取log
        preds_softmax = torch.exp(preds_logsoft)  # softmax

        preds_softmax = preds_softmax.gather(1, labels)  # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1, labels)

        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),
                          preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = loss.mean()

        return loss


class my_loss1(nn.Module):
    def __init__(self):
        super(my_loss1, self).__init__()

        self.criteria1 = torch.nn.CrossEntropyLoss()
        self.criteria2 = torch.nn.MSELoss()

    def forward(self, X, target, inputs, X_AE1, X_AE2, X_AE4):
        loss = calssific_loss_weight * self.criteria1(X, target) + \
               self.criteria2(inputs.float(), X_AE1) + \
               self.criteria2(inputs.float(), X_AE2) + \
               self.criteria2(inputs.float(), X_AE4)
        return loss


class my_loss2(nn.Module):
    def __init__(self):
        super(my_loss2, self).__init__()

        self.criteria1 = focal_loss()
        self.criteria2 = torch.nn.MSELoss()

    def forward(self, X, target, inputs, X_AE1, X_AE2, X_AE4):
        loss = calssific_loss_weight * self.criteria1(X, target) + \
               self.criteria2(inputs.float(), X_AE1) + \
               self.criteria2(inputs.float(), X_AE2) + \
               self.criteria2(inputs.float(), X_AE4)
        return loss


def mixup(x1, x2, y1, y2, alpha):
    beta = np.random.beta(alpha, alpha)
    x = beta * x1 + (1 - beta) * x2
    y = beta * y1 + (1 - beta) * y2
    return x, y


# In[115]:


def BERT_train(model, x_train, y_train, x_test, y_test, event_num):
    model_optimizer = RAdam(model.parameters(), lr=learn_rating, weight_decay=weight_decay_rate)
    model = torch.nn.DataParallel(model)
    model = model.to(device)
# 下面这个预处理,把x_train的后面数据进行了顺序修改,就是把列特征进行了颠倒.让特征可以不依赖顺序.
    x_train = np.vstack(   (x_train, np.hstack(  (x_train[:, len(x_train[0]) // 2:], x_train[:, :len(x_train[0]) // 2])            )       )   )
    y_train = np.hstack((y_train, y_train))
    np.random.seed(seed)
    np.random.shuffle(x_train)
    np.random.seed(seed)
    np.random.shuffle(y_train)

    len_train = len(y_train)
    len_test = len(y_test)
    print("arg train len", len(y_train))
    print("test len", len(y_test))

    train_dataset = DDIDataset(x_train, np.array(y_train))
    test_dataset = DDIDataset(x_test, np.array(y_test))
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(epo_num):
        if epoch < epoch_changeloss:# 小于60轮次之前用loss1. 交叉熵和mse. 看做一个多酚类问题.
            my_loss = my_loss1()
        else:
            my_loss = my_loss2() # 可以加入软标签..........

        running_loss = 0.0

        model.train()
        for batch_idx, data in enumerate(train_loader, 0):
            x, y = data

            lam = np.random.beta(0.5, 0.5)
            index = torch.randperm(x.size()[0]).to(device)
            inputs = lam * x + (1 - lam) * x[index, :] # 样本拼接

            targets_a, targets_b = y, y[index]

            inputs = inputs.to(device)
            targets_a = targets_a.to(device)
            targets_b = targets_b.to(device)

            model_optimizer.zero_grad()
            # forward + backward+update
            X, X_AE1, X_AE2, X_AE4 = model(inputs.float())

            loss = lam * my_loss(X, targets_a, inputs, X_AE1, X_AE2, X_AE4) + (1 - lam) * my_loss(X, targets_b, inputs,
                                                                                                  X_AE1, X_AE2, X_AE4)

            loss.backward()
            model_optimizer.step()
            running_loss += loss.item()

        model.eval()
        testing_loss = 0.0
        with torch.no_grad():#每一个epoch都进行测试
            for batch_idx, data in enumerate(test_loader, 0):
                inputs, target = data

                inputs = inputs.to(device)

                target = target.to(device)

                X, X_AE1, X_AE2, X_AE4 = model(inputs.float())

                loss = my_loss(X, target, inputs, X_AE1, X_AE2, X_AE4)
                testing_loss += loss.item()
        print('epoch [%d] loss: %.6f testing_loss: %.6f ' % (
        epoch + 1, running_loss / len_train, testing_loss / len_test))

    pre_score = np.zeros((0, event_num), dtype=float)
    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader, 0):
            inputs, _ = data
            inputs = inputs.to(device)
            X, _, _, _ = model(inputs.float())
            pre_score = np.vstack((pre_score, F.softmax(X).cpu().numpy()))
    return pre_score


# In[116]:


def roc_aupr_score(y_true, y_score, average="macro"):
    def _binary_roc_aupr_score(y_true, y_score):
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
        return auc(recall, precision)
    def _average_binary_score(binary_metric, y_true, y_score, average):  # y_true= y_one_hot
        if average == "binary":
            return binary_metric(y_true, y_score)
        if average == "micro":
            y_true = y_true.ravel()
            y_score = y_score.ravel()
        if y_true.ndim == 1:
            y_true = y_true.reshape((-1, 1))
        if y_score.ndim == 1:
            y_score = y_score.reshape((-1, 1))
        n_classes = y_score.shape[1]
        score = np.zeros((n_classes,))
        for c in range(n_classes):
            y_true_c = y_true.take([c], axis=1).ravel()
            y_score_c = y_score.take([c], axis=1).ravel()
            score[c] = binary_metric(y_true_c, y_score_c)
        return np.average(score)
    return _average_binary_score(_binary_roc_aupr_score, y_true, y_score, average)
def evaluate(pred_type, pred_score, y_test, event_num):
    all_eval_type = 11
    result_all = np.zeros((all_eval_type, 1), dtype=float)
    each_eval_type = 6
    result_eve = np.zeros((event_num, each_eval_type), dtype=float)
    y_one_hot = label_binarize(y_test, np.arange(event_num))
    pred_one_hot = label_binarize(pred_type, np.arange(event_num))
    result_all[0] = accuracy_score(y_test, pred_type)
    result_all[1] = roc_aupr_score(y_one_hot, pred_score, average='micro')
    result_all[2] = roc_aupr_score(y_one_hot, pred_score, average='macro')
    result_all[3] = roc_auc_score(y_one_hot, pred_score, average='micro')
    result_all[4] = roc_auc_score(y_one_hot, pred_score, average='macro')
    result_all[5] = f1_score(y_test, pred_type, average='micro')
    result_all[6] = f1_score(y_test, pred_type, average='macro')
    result_all[7] = precision_score(y_test, pred_type, average='micro')
    result_all[8] = precision_score(y_test, pred_type, average='macro')
    result_all[9] = recall_score(y_test, pred_type, average='micro')
    result_all[10] = recall_score(y_test, pred_type, average='macro')
    for i in range(event_num):
        result_eve[i, 0] = accuracy_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel())
        result_eve[i, 1] = roc_aupr_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                          average=None)
        result_eve[i, 2] = roc_auc_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                         average=None)
        result_eve[i, 3] = f1_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                    average='binary')
        result_eve[i, 4] = precision_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                           average='binary')
        result_eve[i, 5] = recall_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                        average='binary')
    return [result_all, result_eve]


# In[117]:

#这个是训练代码
def cross_val(feature,label,event_num):
    skf = StratifiedKFold(n_splits=cross_ver_tim)# 先进行cv分解
    y_true = np.array([])
    y_score = np.zeros((0, event_num), dtype=float)
    y_pred = np.array([])
    
    for train_index, test_index in list(skf.split(feature, label))[:1]:#进行数据4:1分割. =======================只跑一个就够累了.
        
        model=BERT(len(feature[0]),bert_n_heads,bert_n_layers,event_num) # 就是提取特征.

        X_train, X_test = feature[train_index], feature[test_index]
        y_train, y_test = label[train_index], label[test_index]
        print("train len", len(y_train))
        print("test len", len(y_test))
        
        pred_score=BERT_train(model,X_train,y_train,X_test,y_test,event_num)
        
        pred_type = np.argmax(pred_score, axis=1)
        y_pred = np.hstack((y_pred, pred_type))
        y_score = np.row_stack((y_score, pred_score))

        y_true = np.hstack((y_true, y_test))
        
    result_all, result_eve= evaluate(y_pred, y_score, y_true, event_num)
    print("打印训练结果")
    print(result_all)
    return result_all, result_eve


# In[118]:


file_path="./"

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bert_n_heads=4
bert_n_layers=4
drop_out_rating=0.3
batch_size=256
len_after_AE=500
learn_rating=0.00001
epo_num=120
cross_ver_tim=5
cov2KerSize=50
cov1KerSize=25
calssific_loss_weight=5
epoch_changeloss=epo_num//2
weight_decay_rate=0.0001
feature_list = ["smile","target","enzyme"]

def save_result(filepath,result_type,result):
    with open(filepath+result_type +'task1'+ '.csv', "w", newline='',encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for i in result:
            writer.writerow(i)
    return 0


# In[119]:


def main():
    #这个数据表示的是论文第三页的data1. 因为df_drug 行数就是572表示572个药.
    conn = sqlite3.connect("./event.db")
    # 使用pandas读取sqlite数据库里面信息.
    df_drug = pd.read_sql('select * from drug;', conn)
    extraction = pd.read_sql('select * from extraction;', conn) # 表示572个药之间的关系
    mechanism = extraction['mechanism']
    action = extraction['action']  # 这个是两种药的作用结果.
    drugA = extraction['drugA'] # a药品
    drugB = extraction['drugB'] # b药品    这3个都是37264长度的数组
    

    
    new_feature, new_label, event_num=prepare(df_drug,feature_list,mechanism,action,drugA,drugB)
    np.random.seed(seed)
    np.random.shuffle(new_feature)
    np.random.seed(seed)
    np.random.shuffle(new_label)
    print("dataset len", len(new_feature))
    
    start=time.time()
    result_all, result_eve=cross_val(new_feature,new_label,event_num)
    print("time used:", (time.time() - start) / 3600)
    save_result(file_path,"all",result_all)
    save_result(file_path,"each",result_eve)


# In[120]:


main()

# epoch [112] loss: 0.007990 testing_loss: 0.002092 
# epoch [113] loss: 0.007801 testing_loss: 0.002266 
# epoch [114] loss: 0.008158 testing_loss: 0.002365 
# epoch [115] loss: 0.008397 testing_loss: 0.002308 
# epoch [116] loss: 0.007760 testing_loss: 0.002102 
# epoch [117] loss: 0.008246 testing_loss: 0.002036 
# epoch [118] loss: 0.008205 testing_loss: 0.002194 
# epoch [119] loss: 0.007798 testing_loss: 0.002063 
# epoch [120] loss: 0.007790 testing_loss: 0.002068 
# 打印训练结果
# [[0.93398631]
#  [0.97748382]
#  [0.94498095]
#  [0.99915725]
#  [0.99466969]
#  [0.93398631]
#  [0.89788755]
#  [0.93398631]
#  [0.9004855 ]
#  [0.93398631]
#  [0.90188248]]
# time used: 0.7421196152104271



# (base) root@e0fd6bdb2fad:~#  cd /AMDE-master-master ; /usr/bin/env /root/miniconda3/bin/python /root/.vscode-server/extensions/ms-python.python-2022.20.2/pythonFiles/lib/python/debugpy/adapter/../../debugpy/launcher 33717 -- /AMDE-master-master/2345/liyang_xiangmu-main/main4.py 
# smile len is:572
# target len is:572
# enzyme len is:572
# dataset len 37264
# train len 29811
# test len 7453
# arg train len 59622
# test len 7453
# epoch [1] loss: 0.069700 testing_loss: 0.041820 
# epoch [2] loss: 0.047169 testing_loss: 0.028944 
# epoch [3] loss: 0.039184 testing_loss: 0.023579 
# epoch [4] loss: 0.035983 testing_loss: 0.020210 
# epoch [5] loss: 0.032743 testing_loss: 0.018172 
# epoch [6] loss: 0.031270 testing_loss: 0.016822 
# epoch [7] loss: 0.029748 testing_loss: 0.015848 
# epoch [8] loss: 0.028850 testing_loss: 0.015159 
# epoch [9] loss: 0.028012 testing_loss: 0.015090 
# epoch [10] loss: 0.026339 testing_loss: 0.012978 
# epoch [11] loss: 0.026175 testing_loss: 0.012674 
# epoch [12] loss: 0.025844 testing_loss: 0.012965 
# epoch [13] loss: 0.024688 testing_loss: 0.011282 
# epoch [14] loss: 0.024334 testing_loss: 0.011083 
# epoch [15] loss: 0.023458 testing_loss: 0.011418 
# epoch [16] loss: 0.022639 testing_loss: 0.011277 
# epoch [17] loss: 0.021963 testing_loss: 0.010204 
# epoch [18] loss: 0.022142 testing_loss: 0.009723 
# epoch [19] loss: 0.021721 testing_loss: 0.010400 
# epoch [20] loss: 0.022604 testing_loss: 0.009255 
# epoch [21] loss: 0.020952 testing_loss: 0.009342 
# epoch [22] loss: 0.021173 testing_loss: 0.009295 
# epoch [23] loss: 0.020076 testing_loss: 0.008991 
# epoch [24] loss: 0.020474 testing_loss: 0.008710 
# epoch [25] loss: 0.019981 testing_loss: 0.008739 
# epoch [26] loss: 0.019770 testing_loss: 0.008215 
# epoch [27] loss: 0.019040 testing_loss: 0.007593 
# epoch [28] loss: 0.018471 testing_loss: 0.007508 
# epoch [29] loss: 0.019193 testing_loss: 0.007845 
# epoch [30] loss: 0.017417 testing_loss: 0.007109 
# epoch [31] loss: 0.018004 testing_loss: 0.006953 
# epoch [32] loss: 0.017742 testing_loss: 0.007493 
# epoch [33] loss: 0.018223 testing_loss: 0.007121 
# epoch [34] loss: 0.018319 testing_loss: 0.006864 
# epoch [35] loss: 0.017090 testing_loss: 0.006879 
# epoch [36] loss: 0.018141 testing_loss: 0.006918 
# epoch [37] loss: 0.018663 testing_loss: 0.007369 
# epoch [38] loss: 0.016906 testing_loss: 0.007021 
# epoch [39] loss: 0.016341 testing_loss: 0.006376 
# epoch [40] loss: 0.017510 testing_loss: 0.006225 
# epoch [41] loss: 0.017482 testing_loss: 0.006330 
# epoch [42] loss: 0.016664 testing_loss: 0.006032 
# epoch [43] loss: 0.016058 testing_loss: 0.006213 
# epoch [44] loss: 0.016861 testing_loss: 0.006332 
# epoch [45] loss: 0.016488 testing_loss: 0.005897 
# epoch [46] loss: 0.016519 testing_loss: 0.005806 
# epoch [47] loss: 0.015836 testing_loss: 0.005606 
# epoch [48] loss: 0.015759 testing_loss: 0.005479 
# epoch [49] loss: 0.016019 testing_loss: 0.005880 
# epoch [50] loss: 0.014943 testing_loss: 0.005777 
# epoch [51] loss: 0.015137 testing_loss: 0.005318 
# epoch [52] loss: 0.015622 testing_loss: 0.005591 
# epoch [53] loss: 0.015334 testing_loss: 0.005302 
# epoch [54] loss: 0.015202 testing_loss: 0.006050 
# epoch [55] loss: 0.015996 testing_loss: 0.005761 
# epoch [56] loss: 0.014447 testing_loss: 0.005459 
# epoch [57] loss: 0.015362 testing_loss: 0.005367 
# epoch [58] loss: 0.015388 testing_loss: 0.005353 
# epoch [59] loss: 0.014624 testing_loss: 0.005287 
# epoch [60] loss: 0.014858 testing_loss: 0.005108 
# epoch [61] loss: 0.009327 testing_loss: 0.002337 
# epoch [62] loss: 0.009513 testing_loss: 0.002469 
# epoch [63] loss: 0.008791 testing_loss: 0.002433 
# epoch [64] loss: 0.009065 testing_loss: 0.002567 
# epoch [65] loss: 0.008977 testing_loss: 0.002383 
# epoch [66] loss: 0.008833 testing_loss: 0.002546 
# epoch [67] loss: 0.009443 testing_loss: 0.002485 
# epoch [68] loss: 0.009476 testing_loss: 0.002290 
# epoch [69] loss: 0.008801 testing_loss: 0.002494 
# epoch [70] loss: 0.008817 testing_loss: 0.002475 
# epoch [71] loss: 0.008559 testing_loss: 0.002482 
# epoch [72] loss: 0.008822 testing_loss: 0.002314 
# epoch [73] loss: 0.009112 testing_loss: 0.002288 
# epoch [74] loss: 0.008914 testing_loss: 0.002103 
# epoch [75] loss: 0.008889 testing_loss: 0.002251 
# epoch [76] loss: 0.008674 testing_loss: 0.002422 
# epoch [77] loss: 0.008535 testing_loss: 0.002197 
# epoch [78] loss: 0.008424 testing_loss: 0.002417 
# epoch [79] loss: 0.008862 testing_loss: 0.002171 
# epoch [80] loss: 0.008302 testing_loss: 0.002446 
# epoch [81] loss: 0.008769 testing_loss: 0.002225 
# epoch [82] loss: 0.007978 testing_loss: 0.002069 
# epoch [83] loss: 0.008511 testing_loss: 0.002302 
# epoch [84] loss: 0.008265 testing_loss: 0.002183 
# epoch [85] loss: 0.008352 testing_loss: 0.002095 
# epoch [86] loss: 0.008166 testing_loss: 0.002355 
# epoch [87] loss: 0.007584 testing_loss: 0.002096 
# epoch [88] loss: 0.008135 testing_loss: 0.002357 
# epoch [89] loss: 0.007974 testing_loss: 0.002181 
# epoch [90] loss: 0.007660 testing_loss: 0.002152 
# epoch [91] loss: 0.008332 testing_loss: 0.002304 
# epoch [92] loss: 0.008066 testing_loss: 0.002170 
# epoch [93] loss: 0.008238 testing_loss: 0.002458 
# epoch [94] loss: 0.007433 testing_loss: 0.002066 
# epoch [95] loss: 0.007964 testing_loss: 0.002141 
# epoch [96] loss: 0.007374 testing_loss: 0.001982 
# epoch [97] loss: 0.007564 testing_loss: 0.001968 
# epoch [98] loss: 0.007723 testing_loss: 0.002017 
# epoch [99] loss: 0.006999 testing_loss: 0.002166 
# epoch [100] loss: 0.007568 testing_loss: 0.002169 
# epoch [101] loss: 0.007047 testing_loss: 0.002057 
# epoch [102] loss: 0.007212 testing_loss: 0.002137 
# epoch [103] loss: 0.007741 testing_loss: 0.002115 
# epoch [104] loss: 0.007746 testing_loss: 0.002134 
# epoch [105] loss: 0.007208 testing_loss: 0.001997 
# epoch [106] loss: 0.007256 testing_loss: 0.001975 
# epoch [107] loss: 0.007416 testing_loss: 0.002249 
# epoch [108] loss: 0.007475 testing_loss: 0.002028 
# epoch [109] loss: 0.007295 testing_loss: 0.002228 
# epoch [110] loss: 0.007082 testing_loss: 0.002165 
# epoch [111] loss: 0.007405 testing_loss: 0.001898 
# epoch [112] loss: 0.007106 testing_loss: 0.001955 
# epoch [113] loss: 0.006897 testing_loss: 0.002052 
# epoch [114] loss: 0.007278 testing_loss: 0.001906 
# epoch [115] loss: 0.007328 testing_loss: 0.001934 
# epoch [116] loss: 0.006838 testing_loss: 0.002040 
# epoch [117] loss: 0.007314 testing_loss: 0.002179 
# epoch [118] loss: 0.007236 testing_loss: 0.001988 
# epoch [119] loss: 0.006897 testing_loss: 0.001926 
# epoch [120] loss: 0.006946 testing_loss: 0.001904 
# 打印训练结果
# [[0.94364685]
#  [0.98268441]
#  [0.94492231]
#  [0.99896033]
#  [0.99743961]
#  [0.94364685]
#  [0.90166515]
#  [0.94364685]
#  [0.92807408]
#  [0.94364685]
#  [0.88597027]]
# time used: 1.6027864358822506