# text2QDMR 项目记录

> by WangYC
>
> @NWPU chang'an Jun.13th - 

## 1. 环境配置

conda env yaml:

```yaml
name: WangYC_env_38
dependencies:
  - _libgcc_mutex=0.1=main
  - _openmp_mutex=4.5=1_gnu
  - blas=1.0=mkl
  - brotlipy=0.7.0=py38h27cfd23_1003
  - bzip2=1.0.8=h7b6447c_0
  - ca-certificates=2022.4.26=h06a4308_0
  - certifi=2021.10.8=py38h06a4308_2
  - cffi=1.15.0=py38hd667e15_1
  - cryptography=37.0.1=py38h9ce1e76_0
  - cudatoolkit=11.3.1=h2bc3f7f_2
  - ffmpeg=4.3=hf484d3e_0
  - freetype=2.11.0=h70c0345_0
  - giflib=5.2.1=h7b6447c_0
  - gmp=6.2.1=h2531618_2
  - gnutls=3.6.15=he1e5248_0
  - idna=3.3=pyhd3eb1b0_0
  - intel-openmp=2021.4.0=h06a4308_3561
  - jpeg=9d=h7f8727e_0
  - lame=3.100=h7b6447c_0
  - lcms2=2.12=h3be6417_0
  - ld_impl_linux-64=2.35.1=h7274673_9
  - libffi=3.3=he6710b0_2
  - libgcc-ng=9.3.0=h5101ec6_17
  - libgomp=9.3.0=h5101ec6_17
  - libiconv=1.15=h63c8f33_5
  - libidn2=2.3.2=h7f8727e_0
  - libpng=1.6.37=hbc83047_0
  - libstdcxx-ng=9.3.0=hd4cf53a_17
  - libtasn1=4.16.0=h27cfd23_0
  - libtiff=4.2.0=h85742a9_0
  - libunistring=0.9.10=h27cfd23_0
  - libuv=1.40.0=h7b6447c_0
  - libwebp=1.2.0=h89dd481_0
  - libwebp-base=1.2.0=h27cfd23_0
  - lz4-c=1.9.3=h295c915_1
  - mkl=2021.4.0=h06a4308_640
  - mkl-service=2.4.0=py38h7f8727e_0
  - mkl_fft=1.3.1=py38hd3c417c_0
  - mkl_random=1.2.2=py38h51133e4_0
  - ncurses=6.3=h7f8727e_2
  - nettle=3.7.3=hbbd107a_1
  - olefile=0.46=pyhd3eb1b0_0
  - openh264=2.1.1=h4ff587b_0
  - openssl=1.1.1o=h7f8727e_0
  - pillow=8.4.0=py38h5aabda8_0
  - pip=21.2.4=py38h06a4308_0
  - pycparser=2.21=pyhd3eb1b0_0
  - pyopenssl=22.0.0=pyhd3eb1b0_0
  - pysocks=1.7.1=py38h06a4308_0
  - python=3.8.5=h7579374_1
  - pytorch=1.11.0=py3.8_cuda11.3_cudnn8.2.0_0
  - pytorch-mutex=1.0=cuda
  - readline=8.1.2=h7f8727e_1
  - requests=2.27.1=pyhd3eb1b0_0
  - setuptools=58.0.4=py38h06a4308_0
  - six=1.16.0=pyhd3eb1b0_0
  - sqlite=3.37.2=hc218d9a_0
  - tk=8.6.11=h1ccaba5_0
  - torchaudio=0.11.0=py38_cu113
  - torchvision=0.12.0=py38_cu113
  - typing_extensions=3.10.0.2=pyh06a4308_0
  - wheel=0.37.1=pyhd3eb1b0_0
  - xz=5.2.5=h7b6447c_0
  - zlib=1.2.11=h7f8727e_4
  - zstd=1.4.9=haebb681_0
  - pip:
    - absl-py==1.0.0
    - blessings==1.7
    - cachetools==5.0.0
    - charset-normalizer==2.0.12
    - click==8.0.4
    - cycler==0.11.0
    - filelock==3.6.0
    - fonttools==4.33.2
    - func-timeout==4.3.5
    - google-auth==2.6.6
    - google-auth-oauthlib==0.4.6
    - gpustat==0.6.0
    - grpcio==1.44.0
    - huggingface-hub==0.4.0
    - importlib-metadata==4.11.3
    - jarowinkler==1.0.2
    - joblib==1.1.0
    - kiwisolver==1.4.2
    - markdown==3.3.6
    - matplotlib==3.5.1
    - nltk==3.7
    - numpy==1.19.5
    - nvidia-ml-py3==7.352.0
    - oauthlib==3.2.0
    - packaging==21.3
    - pandas==1.4.2
    - protobuf==3.20.1
    - psutil==5.9.0
    - pyasn1==0.4.8
    - pyasn1-modules==0.2.8
    - pyparsing==3.0.7
    - python-dateutil==2.8.2
    - pytz==2022.1
    - pyyaml==6.0
    - rank-bm25==0.2.2
    - rapidfuzz==2.0.11
    - regex==2022.3.2
    - requests-oauthlib==1.3.1
    - rsa==4.8
    - sacremoses==0.0.47
    - scikit-learn==1.0.2
    - scipy==1.5.4
    - sentencepiece==0.1.96
    - simcse==0.4
    - sklearn==0.0
    - sql-metadata==2.5.0
    - sqlparse==0.4.2
    - tensorboard==2.8.0
    - tensorboard-data-server==0.6.1
    - tensorboard-plugin-wit==1.8.1
    - threadpoolctl==3.1.0
    - tokenizers==0.11.6
    - torch-tb-profiler==0.4.0
    - tqdm==4.63.0
    - transformers==4.17.0
    - urllib3==1.26.8
    - werkzeug==2.1.1
    - zipp==3.8.0
prefix: /home/lihaoyang/ENTER/envs/python38
```

## 2. text2sql框架

### 2.1 transformers by hugging face 🤗

#### 介绍

Transformers已经在100+种人类语言上提供了32+种预训练语言模型。作为NLP的从业者，真的很难抵制住去一探究竟的诱惑。

以下部分参考关于transformers详细介绍博客：https://zhuanlan.zhihu.com/p/141527015

#### 组件

- Configuration配置类。存储模型和分词器的参数，诸如词表大小，隐层维数，dropout rate等。配置类对深度学习框架是透明的。
- Tokenizer分词器类。每个模型都有对应的分词器，存储token到index的映射，负责每个模型特定的序列编码解码流程，比如BPE(Byte Pair Encoding)，SentencePiece等等。也可以方便地添加特殊token或者调整词表大小，如CLS、SEP等等。
- Model模型类。提供一个基类，实现模型的计算图和编码过程，实现前向传播过程，通过一系列self-attention层直到最后一个隐藏状态层。在最后一层基础上，根据不同的应用会再做些封装，比如XXXForSequenceClassification，XXXForMaskedLM这些派生类。

#### 模型

hugging face维护的模型大全：https://huggingface.co/models

#### 模型使用

````python
from transformers import AutoModel, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
model = AutoModel.from_pretrained('bert-base-chinese')
````

transformers模型管理的方式是为每一个模型起一个唯一的短名，如果一个模型同时有一个配套的tokenizer模型的话，它们会共用一个短名。上面提到的官方中文模型的短名就叫做“bert-base-chinese”。除了bert-base-chinese外，我们还可以找到clue/albert_chinese_small，voidful/albert_chinese_base等等几十个比较热门的社区贡献的中文预训练模型。下面这个网址可以找到所有社区共享由huggingface维护的模型列表。

```python
input_ids = tokenizer.encode('春眠不觉晓', return_tensors='pt')
last_hidden_state, _ = model(input_ids) # shape (1, 7, 768)
v = torch.mean(last_hidden_stat, dim=1) # shape (1, 768)
```

- model()实际上调用的是model.forward()函数
- model(input_ids)返回两个值last_hidden_state以及pooler_output。前者shape=(1, 7, 768)即对输入的sequence中的每个token/字都返回了一个768维的向量。后者是序列中第一个特殊字符[CLS]对应的单个768维的向量，BERT模型会为单句的输入前后加两个特殊字符[CLS]和[SEP]。
- 根据文档的说法，pooler_output向量一般不是很好的句子语义摘要，因此这里采用了torch.mean对last_hidden_state进行了求平均操作

### 2.2 tokenizer

```python
tokenizer = T5TokenizerFast.from_pretrained(
  opt.model_name_or_path,
  add_prefix_space = True
)

if isinstance(tokenizer, T5TokenizerFast):
  tokenizer.add_tokens([AddedToken(" <="), AddedToken(" <"), AddedToken("[FK]")])
```

https://blog.csdn.net/a321123b/article/details/121436837

使用文本的第一步就是将其拆分为单词。单词称为标记（token），将文本拆分为标记的过程称为标记化(tokenization)，而标记化用到的模型或工具称为tokenizer

**T5 tokenizer:**

python transformer库t5:https://huggingface.co/t5-base?text=My+name+is+Wolfgang+and+I+live+in+Berlin

Hugging face T5 源代码：https://huggingface.co/transformers/v4.8.2/_modules/transformers/models/t5/tokenization_t5_fast.html

### 2.3 数据

spider数据集介绍博客：https://juejin.cn/post/7085557671528660999

Spider数据集是一个**多数据库**、**多表**、**单轮**查询的Text-to-SQL数据集。WikiSQL 中查询和表的数量很多，但是所有 SQL 查询都很简单，仅包含 SELECT 和 WHERE 从句，而且WHERE子句中的约束条件不超过3个。此外，每个数据库都只是没有外键的简单的表。在 WikiSQL 上训练的模型在无法处理复杂的 SQL语句要素（如 GROUP BY、ORDER BY 或嵌套查询）和具备多个表和外键的数据库。Spider数据集由 11 名耶鲁大学学生标注，包含 10181 个问题和 5693 个独特的复杂 SQL 查询、200 个具备多个表的数据库，覆盖 138 个不同领域，实际应用性很强。

```json
[
	{
		"question": 语言序列,
		"query": 标准答案,
		"db_id": 用到的数据库id,
		"db_schema":
    [
      {
        "table_name_origin": 原表名,
        "table_name" :表名,
        "column_names_original": ["列1名字","列2名字", ..."列n名字"],
        "column_names": ["列1名字","列2名字", ..."列n名字"],
        "db_contents":[[], [], ... []],
        "column_types": ["列1类型", "列2类型", ... "列n类型"],
      },
      {
        ...表2信息
      },
      {
        ...表n信息
      }
    ],
    "pk":[
      {
        "table_name": "表1名",
        "column_name": "表1主键名"
      },
      {
        "table_name": "表2名",
        "column_name": "表2主键名"
      },
      {
        ...
      }
    ],
    "fk":[
      {
        "source_table_name": "concert",
        "source_column_name": "stadium_id",
        "target_table_name": "stadium",
        "target_column_name": "stadium_id"
      },
      {
        "source_table_name": "singer_in_concert",
        "source_column_name": "singer_id",
        "target_table_name": "singer",
        "target_column_name": "singer_id"
      },
      {
        "source_table_name": "singer_in_concert",
        "source_column_name": "concert_id",
        "target_table_name": "concert",
        "target_column_name": "concert_id"
      }
    ],
]
```

e.g.

```json
[
  {
    "question": "How many singers do we have?",
    "query": "SELECT count(*) FROM singer",
    "db_id": "concert_singer",
    "db_schema": [
      {
        "table_name_original": "singer",
        "table_name": "singer",
        "column_names_original": [
          "name",
          "singer_id",
          "country",
          "age",
          "is_male",
          "song_name",
          "song_release_year"
        ],
        "column_names": [
          "name",
          "singer id",
          "country",
          "age",
          "is male",
          "song name",
          "song release year"
        ],
        "db_contents": [
          [],
          [],
          [],
          [],
          [],
          [],
          []
        ],
        "column_types": [
          "text",
          "number",
          "text",
          "number",
          "others",
          "text",
          "text"
        ]
      },
      {
        "table_name_original": "singer_in_concert",
        "table_name": "singer in concert",
        "column_names_original": [
          "singer_id",
          "concert_id"
        ],
        "column_names": [
          "singer id",
          "concert id"
        ],
        "db_contents": [
          [],
          []
        ],
        "column_types": [
          "text",
          "number"
        ]
      },
      {
        "table_name_original": "stadium",
        "table_name": "stadium",
        "column_names_original": [
          "lowest",
          "location",
          "name",
          "capacity",
          "highest",
          "average",
          "stadium_id"
        ],
        "column_names": [
          "lowest",
          "location",
          "name",
          "capacity",
          "highest",
          "average",
          "stadium id"
        ],
        "db_contents": [
          [],
          [],
          [],
          [],
          [],
          [],
          []
        ],
        "column_types": [
          "number",
          "text",
          "text",
          "number",
          "number",
          "number",
          "number"
        ]
      },
      {
        "table_name_original": "concert",
        "table_name": "concert",
        "column_names_original": [
          "year",
          "theme",
          "concert_id",
          "stadium_id",
          "concert_name"
        ],
        "column_names": [
          "year",
          "theme",
          "concert id",
          "stadium id",
          "concert name"
        ],
        "db_contents": [
          [],
          [],
          [],
          [],
          []
        ],
        "column_types": [
          "text",
          "text",
          "number",
          "text",
          "text"
        ]
      }
    ],
    "pk": [
      {
        "table_name": "stadium",
        "column_name": "stadium_id"
      },
      {
        "table_name": "singer",
        "column_name": "singer_id"
      },
      {
        "table_name": "concert",
        "column_name": "concert_id"
      },
      {
        "table_name": "singer_in_concert",
        "column_name": "concert_id"
      }
    ],
    "fk": [
      {
        "source_table_name": "concert",
        "source_column_name": "stadium_id",
        "target_table_name": "stadium",
        "target_column_name": "stadium_id"
      },
      {
        "source_table_name": "singer_in_concert",
        "source_column_name": "singer_id",
        "target_table_name": "singer",
        "target_column_name": "singer_id"
      },
      {
        "source_table_name": "singer_in_concert",
        "source_column_name": "concert_id",
        "target_table_name": "concert",
        "target_column_name": "concert_id"
      }
    ],
    "table_labels": [
      1,
      0,
      0,
      0
    ],
    "used_tables": [
      "singer"
    ],
    "column_labels": [
      [
        0,
        0,
        0,
        0,
        0,
        0,
        0
      ],
      [
        0,
        0
      ],
      [
        0,
        0,
        0,
        0,
        0,
        0,
        0
      ],
      [
        0,
        0,
        0,
        0,
        0
      ]
    ],
    "used_columns": [
      "*"
    ]
  },
]
```

### 2.4 评估

Evaluation Metrics
根据查询的难度来衡量系统的准确性，伴随语料库一起发布了官方的评测脚本
组件匹配：对SELECT/WHERE/GROUP BY/KEYWORDS这些组件分解成多个子组件，例如SELECT avg(col1), max(col2), min(col1)，首先解析并分解为一个集合(avg, min, col1), (max, col2)，然后再查看是否匹配；在评估中，将每个组件视为一个集合，也就是调整顺序不影响结果，例如SELECT avg(col1), min(col1), max(col2)和SELECT avg(col1), max(col2), min(col1)被认为是相同的查询。对于每个组件的整体性能，采用的是在每个精确集匹配上的F1 score
精确匹配：当且仅当每个组件都正确时，预测的查询才是正确的。
精确匹配可能会导致false negative【不理解】的评估，因此还考虑了执行准确度，同样的如果返回的结果是和标准一样，但语义不同时，可能会出现false positive的报错，这一点也可以彼此互补；最后如果出现了JOIN和GROUP在查询语句中，则评估可以接受多个keys。
为了更好地了解模型在不同查询上的性能，将SQL查询分为了4个级别：简单、中等、困难、特别困难。根据SQL组件的数量、选择和条件来定义难度，包含更多SQL关键字例如GROUP BY、ORDER BY、INTERSECT、嵌套子查询、列选择和聚合语句等会被认为难度是很大的。
————————————————
原文链接：https://blog.csdn.net/qq_45429238/article/details/121498805

### 问题整理

* ~~数据处理里面upper bound的含义？数据处理中各项具体操作含义？~~
* ~~tokenizer使用：~~
  * ~~models和tokenizer可以不一致？~~ 
  * ~~add_tokens是进行了什么操作,为什么先判断isinstance以后再add_tokens？~~
  * ~~具体模型的资料可以在哪查找？https://huggingface.co/t5-base 里面似乎没有详细使用说明，例如`model.resize_token_embeddings(len(tokenizer))`~~
  * 
* ~~这里beamsearch的时候是根据batch来的还是一条数据来的？~~
* ~~模型使用只要id和mask就可以了？这个id的含义是什么？~~
* ~~evaluator里面gold的含义？~~
* ~~metric的思路？~~
* <img src="text2QDMR 项目记录.assets/image-20220613205958854.png" alt="image-20220613205958854" style="zoom:50%;" />
* ~~1卡无负载工作状态是p8？~~

## 3. text2QDMR

### 3.1 需要做的任务

* 把数据中的query替换成QDMR的groundtruth
* beam search改成贪心
* evaluation部分进行修改，尝试用执行结果进行评估分析

