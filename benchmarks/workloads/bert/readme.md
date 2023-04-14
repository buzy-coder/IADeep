#    <center>问答系统SQuAD说明文档</center> 

​																											

## 问题定义

给出一段文本和一个文本相关的问题，由机器给出问题的答案，如果问题的正确答案可以由文本得到，那么需要保证正确答案是在文本中出现过的连续片段，否则问题被判定为不可回答的。

下面给出一个问答的示例：

![img](img\d816e94898a346b4a6c39466d6f72270.jpeg)

#### 利用深度学习模型进行阅读理解

将已经Tokenized的文本+问题序列输入到用于处理序列的模型（RNN，Transformer等）中后，模型的输出是序列中各个token对应的两个概率，一个是该token是答案开头的概率，一个是该token是答案结束的概率。

下面给出利用bert进行阅读理解的示例：

![image-20210626203014098](img\image-20210626203014098.png)



## 数据集介绍

SQuAD(The Stanford Question Answering Dataset) 是由 Rajpurkar 等人 [1] 提出的阅读理解任务数据集。该数据集包含 10 万个（问题，原文，答案）三元组，原文来自于 536 篇维基百科文章，而问题和答案的构建主要是通过众包的方式，让标注人员提出最多 5 个基于文章内容的问题并提供正确答案，且答案出现在原文中。在SQuAD的基础上， Rajpurkar 等人 [2]新增了一项任务：判断一个问题能否根据提供的阅读文本作答，即要求机器能够判断出不可回答的问题，新的数据集称为SQuAD2.0.

下面给出一个SQuAD2.0的文本+问题+答案的示例：

```
{
 'title': 'Beyoncé',
 'paragraphs': [
 				{
    'qas': [{
        'question': 'When did Beyonce start becoming popular?',
        'id': '56be85543aeaaa14008c9063',
     	'answers': [{'text': 'in the late 1990s', 'answer_start': 269}],'is_impossible': False}],
        'context':'Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny\'s Child. Managed by her father, Mathew Knowles, the group became one of the world\'s best-selling girl groups of all time. Their hiatus saw the release of Beyoncé\'s debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles "Crazy in Love" and "Baby Boy".'}]
}

```

**is_impossible** 表示该问题是否是不可回答的，**answer_start** 表示该答案在原文序列中的开始的位置。

## 数据集处理

```python
class SquadDataset(data.Dataset):
    def __init__(self, split = 'train'):
        self.split = split
        if split == 'train':
            self.datapath = 'data/train-v2.0.json'
        elif split == "valid":
            self.datapath = "data/dev-v2.0.json"
        
        self.tokenizer =  BertTokenizerFast.from_pretrained('bert-base-uncased')

        contexts, questions, answers = read_squad(self.datapath)
        adjust_start_idx(contexts, answers)
        self.encodings = self.tokenizer(contexts, questions, truncation=True, padding=True)
        cal_token_positions(self.encodings, answers)

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)
```

本实验利用了Pytorch中的torch.utils.data.Dataset建立了数据集SquadDataset。在SquadDataset初始化的时候，首先利用**read_squad**函数将数据从json文件中读出来,然后考虑到原始数据集中提供的**answer_start** 和真实结果可能有一到两个词的误差，即可能**answer_start - 1** 或**answer_start - 2**才是真正的 **answers_text**的位置，所以利用**adjust_start_idx**来调整**answer_start**的位置。然后利用 huggingface 提供的tokenizer来进行tokenization,即把文本转换成数字id,最后把**answer_position**由字符位置转换到token位置并加入到**encoding**里。需要注意的是，SQuAD2.0中有许多没有答案的问题，本实验将不可回答的问题的**answer_start** 和**answer_end** 都设置为self.tokenizer.model_max_length - 1. 用于数据集处理的代码详见**dataset.py**.

## 模型选择

本实验选择利用预训练模型Bert[3]作为初始模型，并在bert后添加一个线性层和一个softmax层，来估计文本序列中单词对应的开始概率和结束概率。

估计开始概率的示意图如下：

![image-20210626223108817](img\image-20210626223108817.png)

考虑到SQuAD2.0的特性，为了能够判断出问题是否可回答，本实验又在上述模型的基础上添加了一个线性分类器，即用第一个token([SEP])的输出连接一个线性层，得到该问题是否可回答的一个二分类label，模型示意图如下所示：

![image-20210628020711385](img\image-20210628020711385.png)

本实验在 huggingface 提供的BertModel的基础上实现了BertForSQuAD，详见**model.py**.

BertForSQuAD的输入分为五个部分：

1. ​    input_ids: 已经经过tokenization的文本 + 问题序列。
2. ​    attention_mask: 由于我们的模型要求输入的序列长度固定，所以我们需要对长度小于输入序列的文本进行padding, attention_mask是一个[0,1]序列，它表明了输入序列中哪些token是padding得到的，这部分序列在进行attention计算的时候需要被mask掉（**attention_mask** = 0).
3. ​    token_type_ids: [0,1]序列，用来区分序列中的文本部分和问题部分。
4. ​    start_positions：[0,1]序列，用来区分序列中的token是否为answer_start.
5. ​    end_positions: [0,1]序列，用来区分序列中的token是否为answer_end.
6. ​     ans_exists: 0或1，0表示答案不存在，1表示答案存在。

考虑到bert能处理的最长token序列仅为512，超过512的就要将其截断，这不符合阅读理解任务需要对长文本进行建模的需要，所以本实验又选择了能够处理长文本序列的XLNet[4]作为初始模型，并将其结果和bert进行比较。本实验在 huggingface 提供的XLNetModel的基础上实现了XLNetForSQuAD，详见**model.py**.

## 实验结果

本实验选择huggingface 中提供的BertTokenizerFast和XLNetTokenizerFast来分别对Bert和XLNet的输入做Tokenization.本实验选择AdamW作为optimizer并选择linear_warmup作为学习率调整策略，具体的参数详见**base_options.py**. 本实验的训练部分详见**train.py**.

为了进行模型训练，可以在命令行输入：

```
python train.py --model [bert or xlnet]
```

其它的训练参数详见**base_option.py**.

下图是BertForSQuAD训练过程中验证集loss（loss function为torch.nn.Crossentropy）和训练集loss的变化过程：

![image-20210627225332113](img\image-20210627225332113.png)

![image-20210627225343971](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20210627225343971.png)

可以看到，随着训练过程的推进，测试集的loss是先下降后上升的，上升的原因可能是模型对训练集过拟合，而训练集的loss则是一直下降。

BertForSQuAD在验证集上的结果：

```
{'exact_match': 0.6625115808978354, 'f1': 0.7108461173046997}
```

其中**exact_match**表示输出结果和真实候选结果中的其中一个答案完全一样的比例，若模型预测结果和真实候选结果均为无答案，则exact_match也为1，**f1  score**则表示输出结果和真实候选结果中最接近输出结果的答案之间的单词匹配准确度。

BertForSQuAD在验证集上的结果存储在 results/SQuAD_bert_pred_result.json 中，为了验证上述结果，可以在命令行输入：

```
python eval.py
```





## 环境需求

本实验需要pytorch >= 1.6 以及 transformers >= 4.0. 如果需要开启训练过程可视化选项（关闭该选项需要在命令行中输入python train.py --visualize False），则需要tensorboardX>=0.4.0.

## 参考文献

[1] Rajpurkar P, Zhang J, Lopyrev K, et al. SQuAD: 100, 000+ Questions for Machine Comprehension of Text[C]//EMNLP. 2016.

[2] Rajpurkar P, Jia R, Liang P. Know What You Don’t Know: Unanswerable Questions for SQuAD[C]//Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers). 2018: 784-789.

[3] Devlin J, Chang M W, Lee K, et al. Bert: Pre-training of deep bidirectional transformers for language understanding[J]. arXiv preprint arXiv:1810.04805, 2018.

[4] Yang Z, Dai Z, Yang Y, et al. Xlnet: Generalized autoregressive pretraining for language understanding[J]. arXiv preprint arXiv:1906.08237, 2019.

