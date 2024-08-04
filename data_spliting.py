import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold


if __name__ == '__main__':
    origin_data_filename = "merged_data.csv"
    # 使用 pandas 的 read_csv 函数读取CSV文件，并将数据存储在名为 data 的 DataFrame 中
    data = pd.read_csv(origin_data_filename)
    labels = data['label']
    #缺失值填0
    data = data.fillna(0)
    #划分训练集和验证+测试集
    data_next, data_train, label_next, label_train = train_test_split(data, labels, test_size=100000, stratify=labels)
    #划分验证集和测试集
    labels = data_next['label']
    data_test, data_val, label_test, label_val = train_test_split(data_next, labels, test_size=20000, stratify=labels)
    # 将特征和标签合并
    data_train['label'] = label_train
    data_val['label'] = label_val
    data_test['label'] = label_test

    # 保存数据集
    data_train.to_csv('splited_data/train_data.csv', index=False)
    data_val.to_csv('splited_data/val_data.csv', index=False)
    data_test.to_csv('splited_data/test_data.csv', index=False)