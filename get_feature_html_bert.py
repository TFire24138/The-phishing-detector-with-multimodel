import torch
from transformers import BertTokenizer, BertModel
from bs4 import BeautifulSoup
import requests
from html import unescape
import os
import os.path as osp
import numpy as np


#指定本地模型和分词器的路径（确保这是一个目录路径）
local_model_dir = r'E:\大学作业\大创\模型\uncased_L-12_H-768_A-12\uncased_L-12_H-768_A-12'  # 使用原始字符串
#从本地加载分词器
tokenizer = BertTokenizer.from_pretrained(local_model_dir)
# 从本地加载模型
model = BertModel.from_pretrained(local_model_dir)


def extract_features_from_html(html_content):
    # 解析HTML内容并提取文本
    try:
        # 只使用html.parser解析器
        soup = BeautifulSoup(html_content, "html.parser")
        text = soup.get_text()
        # 解码HTML实体
        text = unescape(text)
        # 分词
        tokens = tokenizer.encode(text, add_special_tokens=True)
        tokens = tokens[:512]  # 确保输入在模型的最大长度内
        # 转换为PyTorch张量
        input_ids = torch.tensor(tokens).unsqueeze(0)  # 确保形状为 (1, seq_len)
        # 前向传播，获取隐藏状态
        with torch.no_grad():
            outputs = model(input_ids)
        # 获取最后的隐藏状态
        last_hidden_states = outputs.last_hidden_state
        # 使用平均池化
        mean_pooling = last_hidden_states.mean(dim=1)  # (1, 768)
        return mean_pooling
    except Exception as e:
        return None  # Skip this HTML content on any error

# Example HTML content
html_path = r"E:\大学作业\大创\模型\output"
html_list = os.listdir(html_path)
print(html_list)
# Extract features

file_count = 0     # 用于计数提取的HTML文件数

feature_arr = []
for tmp_html in html_list:
    with open(osp.join(html_path, tmp_html), 'r', encoding='utf-8') as file:
        html_content = file.read()
    features = extract_features_from_html(html_content)
    if features is not None:
        file_count += 1
        print(f"已提取 {file_count} 个 HTML 文件的特征。")
        #print('tmp feature:{}'.format(features.shape))
        feature_arr.append(features)
    else:
        print(f"提取失败")
feature_arr = torch.cat(feature_arr)

indices = [int(os.path.splitext(fname)[0].split('_')[-1]) for fname in html_list]

# 将特征向量和索引一起排序
sorted_features = [f for _, f in sorted(zip(indices, feature_arr))]

# 将排序后的特征向量堆叠成一个二维数组
features_np = np.stack(sorted_features)

# 保存排序后的特征数组和索引到 .npz 文件
np.savez_compressed(
    r'E:\大学作业\大创\模型\sorted_features.npz',
    features=features_np,
    indices=indices  # 这里添加索引的保存
)

print(f"特征向量和索引已保存到 'sorted_features.npz'，索引为: {indices}")


# # 加载 .npz 文件
# with np.load(r'E:\大学作业\大创\模型\sorted_features.npz') as npz:
#     features_np = npz['features']
#     indices = npz['indices']
#
# # 将 NumPy 数组转换回 PyTorch 张量
# features_torch = torch.from_numpy(features_np)
#
# print("已加载排序后的特征:")
# print(f"特征张量的形状: {features_torch.shape}")
#
# # 打印每个特征向量对应的索引
# print("每个特征向量对应的索引:")
# for idx in indices:
#     print(idx)

