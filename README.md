# pytorch_bert_crf
本项目是基于pytorch的bert_crf命名实体识别

## 相关说明
```
--checkpoints: 保存模型和加载模型
--data: 存放数据和数据初步处理
--logs: 存放日志
--model: 存放模型、评估函数
--model_hub: 存放预训练模型bert
--utils: 解码函数和辅助函数
--config.py: 相关配置文件
--data_loader.py: 处理模型所需的输入数据格式
--inference.py: 模型推理
--run.py: 训练和测试的运行文件
--trainer.py: 训练和测试封装的类
```
在hugging face上预先下载好预训练的bert模型，存放在model_hub文件夹下

## 一般步骤
1. data下新建CNER文件夹，将原始数据存放在该文件夹下。通过preprocess_raw_data.py，生成相关文件，包括labels.json、test.json、train.json等。
2. 调节config.py的配置文件，训练时，确保参数do_train=True，如果要看测试结果，可do_test=True。若训练完之后，再看测试结果，即do_train=False和do_test=True，那么要确保加载的模型路径resume参数不能为None。
3. 运行inference.py，可以看到单条数据推理结果。

## 效果展示：
```
2022-10-18 22:00:01,724 - [Test] loss：0.913140 precision：0.9404 recall：0.9272 f1-score：0.9337
          precision    recall  f1-score   support

    NAME       1.00      0.98      0.99       110
    RACE       1.00      0.93      0.97        15
   TITLE       0.93      0.92      0.93       690
     ORG       0.94      0.90      0.92       523
     PRO       0.82      1.00      0.90        18
     LOC       1.00      1.00      1.00         2
     EDU       0.95      0.99      0.97       106
    CONT       1.00      1.00      1.00        33

micro-f1       0.94      0.93      0.93      1497
```
## 推理结果
```
 输入：
    text = '1963年出生，工科学士，高级工程师，北京物资学院客座副教授。'
 输出：
    {'文本': '1963年出生，工科学士，高级工程师，北京物资学院客座副教授。',
     '预测实体': {
            'EDU': [('学士', 10)],
            'ORG': [('北京物资学院', 19)],
            'PRO': [('工科', 8)],
            'TITLE': [('高级工程师', 13), ('客座副教授', 25)]
            }}
```