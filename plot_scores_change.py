#!/usr/local/bin python
# -*- coding: utf-8 -*-

# Created on 202106242058
# Author:     zhuoyin94 <zhuoyin94@163.com>
# Github:     https://github.com/MichaelYin1994

'''
本模块(plot_score_change.py)绘制赛程期间主要分数变化情况。
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':

    # 提交的基础信息
    col_names = ['sub_date', 'val_score', 'online_score', "description"]
    score_list = [['2021-06-24 21:30:00', 0.62, 0.44, '第一次正常提交分数，没有采用任何数据增强方法。'],
                  ['2021-06-24 22:30:00', 0.67, 0.45, '正确使用了random flip与random raotation']]

    # 分数数据预处理
    df = pd.DataFrame(score_list, columns=col_names)
    df['sub_date'] = pd.to_datetime(df['sub_date'])

    # 分数随时间变化图
    # ---------------------
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df['val_score'].values, linestyle="--", marker="s", color="k",
            linewidth=2, markersize=4.5, label='validation')
    ax.plot(df['online_score'].values, linestyle="-", marker="o", color="b",
            linewidth=2, markersize=4.5, label='online_testing')

    ax.grid(True)
    # ax.set_xlim(0.6, )
    ax.set_ylim(0, )
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Top-1 Accuracy", fontsize=10)
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(fontsize=10)
    plt.tight_layout()