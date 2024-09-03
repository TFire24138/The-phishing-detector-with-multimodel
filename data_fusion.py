import numpy as np
import pandas as pd

# html_data = np.load("sorted_features.npz")
# array1 = html_data['features']
# array2 = html_data['indices']
# df1 = pd.DataFrame(array1)
# df2 = pd.DataFrame(array2)
# html_data= pd.concat([df1, df2], axis=1, ignore_index=True)
# html_data.index = html_data.iloc[:,768]  #将索引更新为对应的号

#读取PCA后的html特征
html_data_path = "PCA_HTMLfeatures.csv"
html_data = pd.read_csv(html_data_path)
html_data = html_data.set_index(html_data.columns[0]) #重新设置行索引

# 读取url特征，预处理
url_csv_path = "URLfeatures.csv"
url_data = pd.read_csv(url_csv_path)
# 由于URLfeatures.csv的索引是从0开始的，而html.csv中的index是从1开始的，
# 我们需要将url_data的索引加1，以便与html.csv中的index对齐
url_data.index = url_data.index + 1
# 确保url_data的索引列是整数类型
url_data.index = url_data.index.astype(int)

# 使用merge函数根据index列合并两个数据集，只合并行索引相匹配的行
merged_data = pd.merge(url_data,html_data, how='inner', left_index=True, right_index=True)

# 如果列名为URL的列存在，将列名作为字符串传递给columns参数
columns_to_drop = ['URL']
merged_data.drop(columns=columns_to_drop, inplace=True)

# 打印合并后的数据
print(merged_data)
# 如果需要，将合并后的数据保存到新的CSV文件中
merged_data.to_csv('merged_data.csv', index=True)
print("合并完成，结果已保存到 'merged_data.csv'")



# 要将数据转为numpy数组的形式，在喂给模型的的时候以这种形式。⭐
# merged_data_numpy = merged_data.values
