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
    