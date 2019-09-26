# U-Net_CGAN-with-Keras
##
## 去除了原先U-Net当中的Dropout layer
## 使得在自己的训练集上也可以进行语义分割
##
## 可以考虑使用LeakyReLU作为卷积层的激活函数 
## 最终的输出使用tanh 归一到-1~1之间的输出
##
## feature visual.py 用于查看中间层的输出
