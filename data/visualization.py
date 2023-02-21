'''
Author: Yuyang Zhou @ Westlake University
Date: 2023-02-19 11:29:13
LastEditTime: 2023-02-19 15:24:03
LastEditors: Please set LastEditors
Description: 
'''

import matplotlib.pyplot as plt
import seaborn as sns

def distribution(data):
    plt.style.use('_mpl-gallery')
    plt.figure(figsize=(3,2))
    sns.distplot(data)
    plt.legend()
    plt.grid(linestyle='--')
    plt.show()

def roc_graph(fpr, tpr, auc):
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % auc)
    plt.legend(loc = 'lower right')
    plt.plot([0,1], [0,1], 'r--')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()