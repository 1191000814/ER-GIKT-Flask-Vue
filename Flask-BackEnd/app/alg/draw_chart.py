"""
绘制图表
"""
import os.path

import matplotlib.pyplot as plt
import numpy as np

data_name = '2023_05_17#19_26_12'

plt.title('the loss, acc and auc of GIKT model')
plt.xlabel('epoch')
plt.ylabel('value')

y = np.load(os.path.join('chart_data', data_name + '.txt'))
x = np.arange(y.shape[1])
y_loss, y_acc, y_auc = y[0], y[1], y[2]
plt.plot(x, y_loss, label='loss')
plt.plot(x, y_acc, label='acc')
plt.plot(x, y_auc, label='auc')
plt.legend()

plt.savefig(os.path.join('chart_img', data_name + 'jpg'))
plt.show()