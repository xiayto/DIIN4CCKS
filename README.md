# 1、目录结构：
* 将数据存放于data之中，train.txt是训练数据，dev.txt是要预测的数据
* data中还要存放训练好的字向量wx_vector_char.pkl
	预处理的结果会放在./data/train和./data/dev中
* 训练过程中会保存模型到models里面
* 最后的预测结果是predcit.csv

# 2、执行顺序：
## 2.1 先执行python preprocess.py
* 对数据进行预处理
* 首先是统计train和predict出现过的字，做成字典与字的word2vector向量对应（每一个字变成数字，数字与词向量矩阵的该字的顺序对应）
* 然后提取两个的特征：
	* 1 chars，提取的是改字附近的n个字（用于后面做卷积处理拼接在该字的w2v向量字后）
	* 2 excat match，该字是否在另一个句子中有出现（用于拼接在w2v向量之后，有为1，没有为0）
* 比较重要的参数：
	* --p --h 第一个句子和第二个句子的长度，设为60，即是小于60个字补0，大于60个字截断后面
	* --chars_per_word 考虑该字前后n个字，如果是4，考虑前4个和该字和后三个，一共8个

## 2.2 在执行python train.py
参数介绍：

* --batch_size :	设置过大可能会显存不足，设为整数可能会显存分配错误（keras的问题）
* --char_conv_kernel：处理前后几个字卷积核的大小
* --omit_chars：是否不考虑前后几个字的卷积特征，1为不考虑
* --omit_exact_match：是否不考虑该字是否在另一句出现，1为不考虑
* --is_train：是训练还是测试，1为train 0为预测

## 2.3 
* 执行python train.py --is_train 0
* 在models中选出最好的模型，进行预测，结果保存在当前目录的predict.csv中

# 3、可调节的参数介绍：
* 1、对p和h的长度，根据数据集的大小调节p和h的长度
* 2、chars范围的调节
* 3、dropout decay rate 和 dropout decay interval
（本文训练开始的keep rate为1，随着训练dropout变大减缓过拟合，每dropout decay interval步乘以decay rate）
* 4、l2 full step \ l2 full ratio（ 同样，本文的参数加入了l2的正则项，正则项随训练增大增大减轻过拟合）
	* l2 full step：通过增大总步长降低变大的速度
	* l2 full ratio：最大值时的l2项大小

# 4、实验结果

数据集规模：train 10万条，predict 1万条

* 平均长度：11.9
* 最长句子：153
* 长度>50：799
      >60：350
      >70：168


## 4.1 最优结果：89.72%
最优结果参数设置：
两个句子的长度：60，不加入exact_match特征
## 4.2 句子长度的影响
|句子长度|在10000个数据上做预测的时间|最优结果|
| :------: | :------: | :------: |
|   50 |   29s | 88.35% |
|   60 |   40s | 89.72% |
|   70 |   60s | 88.62% |

## 4.3 特征对模型的影响

| 模型特征 | 最优结果 |
| :------:| :------:| 
| W2V，Chars | 89.72% | 
| W2V，exact match| 86.45% |
| W2V，exact match, chars| 89.41% |

