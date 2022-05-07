# *_* coding : utf-8 *_*
# Author   : Zhuosd
# Time     : 23/02/2022 10:02 PM
# Filename : loaddatasets.py
# Product  : PyCharm
import random
import pandas as pd
import numpy as np
import scipy.io as scio
from sklearn.model_selection import train_test_split

class LoadData():
    def __init__(self, path, name):
        self._path = path
        self._name = name
        self.precent = 0.12 # 控制初试index的数量

    def load_data(self):
        if self._name == "a8a":
            print("The Datasets is ", self._name)
            a8a = pd.read_csv(self._path + "/MaskData/a8a/X_imp.txt", header = None, sep = " ")
            a8a_y = pd.read_csv(self._path + "/DataLabel/a8a/Y_label.txt", header = None)
            # X_train, X_test, Y_train, Y_test = train_test_split(a8a, a8a_y, test_size=0.2, shuffle=True)
            X_train = a8a
            Y_train = a8a_y
            initial_label = random.sample(range(0,len(Y_train)),int(len(Y_train) * self.precent)) #Y_train
            initial_label.sort
            # K = len(a8a[0].unique())
            X_train = X_train.values
            return X_train, Y_train, initial_label# X_train, X_test, Y_train, Y_test, initial_label

        elif self._name == "trapezoid":
            print("The Datasets is ", self._name)
            australian = pd.read_csv(self._path + "/MaskData/trapezoid/X_imp.txt", header = None, sep = " ")
            australian_y = pd.read_csv(self._path + "/DataLabel/trapezoid/Y_label.txt", header = None)
            X_train = australian
            Y_train = australian_y
            initial_label = random.sample(range(0, len(Y_train)), int(len(Y_train) * self.precent))  # Y_train
            initial_label.sort
            X_train = X_train.values
            return X_train, Y_train, initial_label

        elif self._name == "trapezoid":
            print("The Datasets is ", self._name)
            credit = pd.read_csv(self._path + "/MaskData/trapezoid/X_imp.txt", header = None, sep = " ")
            credit_y = pd.read_csv(self._path + "/DataLabel/trapezoid/Y_label.txt", header = None)
            X_train = credit
            Y_train = credit_y
            initial_label = random.sample(range(0, len(Y_train)), int(len(Y_train) * self.precent))  # Y_train
            initial_label.sort
            X_train = X_train.values
            return X_train, Y_train, initial_label

        elif self._name == "trapezoid":
            print("The Datasets is ", self._name)
            diabetes = pd.read_csv(self._path + "/MaskData/trapezoid/X_imp.txt", header = None, sep = " ")
            diabetes_y = pd.read_csv(self._path + "/DataLabel/trapezoid/Y_label.txt", header = None)
            X_train = diabetes
            Y_train = diabetes_y
            initial_label = random.sample(range(0, len(Y_train)), int(len(Y_train) * self.precent))  # Y_train
            initial_label.sort
            initial_label = np.array(initial_label)
            X_train = X_train.values
            return X_train, Y_train, initial_label

        elif self._name == "trapezoid":
            print("The Datasets is ", self._name)
            dna = pd.read_csv(self._path + "/MaskData/trapezoid/X_imp.txt", header = None, sep = " ")
            dna_y = pd.read_csv(self._path + "/DataLabel/trapezoid/Y_label.txt", header = None)
            X_train = dna
            Y_train = dna_y
            initial_label = random.sample(range(0, len(Y_train)), int(len(Y_train) * self.precent))  # Y_train
            initial_label.sort
            X_train = X_train.values
            return X_train, Y_train, initial_label

        elif self._name == "trapezoid":
            print("The Datasets is ", self._name)
            german = pd.read_csv(self._path + "/MaskData/trapezoid/X_imp.txt", header = None, sep = " ")
            german_y = pd.read_csv(self._path + "/DataLabel/trapezoid/Y_label.txt", header = None)
            X_train = german
            Y_train = german_y
            initial_label = random.sample(range(0, len(Y_train)), int(len(Y_train) * self.precent))  # Y_train
            initial_label.sort
            X_train = X_train.values
            return X_train, Y_train, initial_label

        elif self._name == "hapt":
            print("The Datasets is ", self._name)
            hapt = pd.read_csv(self._path + "/MaskData/hapt/X_imp.txt", header = None, sep = " ")
            hapt_y = pd.read_csv(self._path + "/DataLabel/hapt/Y_label.txt", header = None)
            X_train = hapt
            Y_train = hapt_y
            initial_label = random.sample(range(0, len(Y_train)), int(len(Y_train) * self.precent))  # Y_train
            initial_label.sort
            X_train = X_train.values
            return X_train, Y_train, initial_label

        # IMDB数据集过来，处理不过来
        # elif self._name == "imdb":
        #     hapt = pd.read_csv(self._path + "/MaskData/hapt/X_imp.txt", header = None, sep = " ")
        #     hapt_y = pd.read_csv(self._path + "/DataLabel/hapt/Y_label.txt", header = None)
        #     X_train = hapt
        #     Y_train = hapt_y
        #     initial_label = random.sample(range(0, len(Y_train)), int(len(Y_train) * self.precent))  # Y_train
        #     initial_label = initial_label.sort
        #     X_train = X_train.values
        #     return X_train, Y_train, initial_label

        elif self._name == "trapezoid":
            print("The Datasets is ", self._name)
            ionosphere = pd.read_csv(self._path + "/MaskData/trapezoid/X_imp.txt", header = None, sep = " ")
            ionosphere_y = pd.read_csv(self._path + "/DataLabel/trapezoid/Y_label.txt", header = None)
            X_train = ionosphere
            Y_train = ionosphere_y
            initial_label = random.sample(range(0, len(Y_train)), int(len(Y_train) * self.precent))  # Y_train
            initial_label.sort
            X_train = X_train.values
            return X_train, Y_train, initial_label

        elif self._name == "kr_vs_kp":
            print("The Datasets is ", self._name)
            kr_vs_kp = pd.read_csv(self._path + "/MaskData/kr_vs_kp/X_imp.txt", header=None, sep=" ")
            kr_vs_kp_y = pd.read_csv(self._path + "/DataLabel/kr_vs_kp/Y_label.txt", header=None)
            X_train = kr_vs_kp
            Y_train = kr_vs_kp_y
            initial_label = random.sample(range(0, len(Y_train)), int(len(Y_train) * self.precent))  # Y_train
            initial_label.sort
            X_train = X_train.values
            return X_train, Y_train, initial_label

        elif self._name == "trapezoid":
            print("The Datasets is ", self._name)
            magic04 = pd.read_csv(self._path + "/MaskData/trapezoid/X_imp.txt", header=None, sep=" ")
            magic04_y = pd.read_csv(self._path + "/DataLabel/trapezoid/Y_label.txt", header=None)
            X_train = magic04
            Y_train = magic04_y
            initial_label = random.sample(range(0, len(Y_train)), int(len(Y_train) * self.precent))  # Y_train
            initial_label.sort
            X_train = X_train.values
            return X_train, Y_train, initial_label

        elif self._name == "trapezoid":
            print("The Datasets is ", self._name)
            splice = pd.read_csv(self._path + "/MaskData/trapezoid/X_imp.txt", header=None, sep=" ")
            splice_y = pd.read_csv(self._path + "/DataLabel/trapezoid/Y_label.txt", header=None)
            X_train = splice
            Y_train = splice_y
            initial_label = random.sample(range(0, len(Y_train)), int(len(Y_train) * self.precent))  # Y_train
            initial_label.sort
            X_train = X_train.values
            return X_train, Y_train, initial_label

        elif self._name == "trapezoid":
            print("The Datasets is ", self._name)
            svmguide3 = pd.read_csv(self._path + "/MaskData/trapezoid/X_imp.txt", header=None, sep=" ")
            svmguide3_y = pd.read_csv(self._path + "/DataLabel/trapezoid/Y_label.txt", header=None)
            X_train = svmguide3
            Y_train = svmguide3_y
            initial_label = random.sample(range(0, len(Y_train)), int(len(Y_train) * self.precent))  # Y_train
            initial_label.sort
            X_train = X_train.values
            return X_train, Y_train, initial_label

        elif self._name == "trapezoid":
            print("The Datasets is ", self._name)
            wbc = pd.read_csv(self._path + "/MaskData/trapezoid/X_imp.txt", header=None, sep=" ")
            wbc_y = pd.read_csv(self._path + "/DataLabel/trapezoid/Y_label.txt", header=None)
            X_train = wbc
            Y_train = wbc_y
            initial_label = random.sample(range(0, len(Y_train)), int(len(Y_train) * self.precent))  # Y_train
            initial_label.sort
            X_train = X_train.values
            return X_train, Y_train, initial_label

        elif self._name == "trapezoid":
            print("The Datasets is ", self._name)
            wdbc = pd.read_csv(self._path + "/MaskData/trapezoid/X_imp.txt", header=None, sep=" ")
            wdbc_y = pd.read_csv(self._path + "/DataLabel/trapezoid/Y_label.txt", header=None)
            X_train = wdbc
            Y_train = wdbc_y
            initial_label = random.sample(range(0, len(Y_train)), int(len(Y_train) * self.precent))  # Y_train
            initial_label.sort
            X_train = X_train.values
            return X_train, Y_train, initial_label

        elif self._name == "wpbc":
            print("The Datasets is ", self._name)
            wpbc = pd.read_csv(self._path + "/MaskData/wpbc/X_imp.txt", header=None, sep=" ")
            wpbc_y = pd.read_csv(self._path + "/DataLabel/wpbc/Y_label.txt", header=None)
            X_train = wpbc
            Y_train = wpbc_y
            initial_label = random.sample(range(0, len(Y_train)), int(len(Y_train) * self.precent))  # Y_train
            initial_label.sort
            X_train = X_train.values
            return X_train, Y_train, initial_label

        # 用于测试matlab数据集
        elif self._name == "mat":
            print("The Datasets is ", self._name)
            matpath = self._path  # "./IRIS.mat"
            data = scio.loadmat(matpath)
            test = data['test']
            initial_label = data['initial_label']
            train = data['train']
            label_train = data['label_train']
            label_test = data['label_test']
            return train, label_train, initial_label
