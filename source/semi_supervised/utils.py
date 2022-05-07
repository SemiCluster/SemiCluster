import pandas as pd
import scipy.io as scio

def read_mat_file(path):
    matpath = path
    data = scio.loadmat(matpath)
    test          = data['test']
    initial_label = data['initial_label']
    train         = data['train']
    label_train   = data['label_train']
    label_test    = data['label_test']
    return test, initial_label, train, label_train, label_test

def read_txt_file(path):
    txtfile = path
    data = pd.read