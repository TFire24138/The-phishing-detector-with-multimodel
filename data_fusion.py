import numpy as np
import pandas as pd

# 使用np.load读取html特征
html_data = np.load("sorted_features.npz")
# 假设.npz文件中包含两个数组，分别命名为'array1'和'array2'
array1 = html_data['features']
array2 = html_data['indices']
# 将NumPy数组转换为Pandas DataFrame
df1 = pd.DataFrame(array1)
df2 = pd.DataFrame(array2)
html_data= pd.concat([df1, df2], axis=1, ignore_index=True)
html_data.index = html_data.iloc[:, 768]  #将索引更新为对应的号

#读取url特征，预处理
url_csv_path = "URLfeatures.csv"
url_data = pd.read_csv(url_csv_path)
# URL的索引是从0开始的，而html的index是从1开始的，需要将url的索引加1以便与html中的index对齐
url_data.index = url_data.index + 1
# 确保url_data的索引列是整数类型
url_data.index = url_data.index.astype(int)

# 使用merge函数根据index列合并两个数据集，只合并行索引相匹配的行
merged_data = pd.merge(url_data,html_data, how='inner', left_index=True, right_index=True)

column_to_drop_by_index = merged_data.columns[855]
# 然后，使用drop方法删除第769列和列名为URL的列
# 如果列名为URL的列存在，将列名作为字符串传递给columns参数
columns_to_drop = [column_to_drop_by_index, 'URL']
merged_data.drop(columns=columns_to_drop, inplace=True)

# 打印合并后的数据
print(merged_data)
# 如果需要，将合并后的数据保存到新的CSV文件中
merged_data.to_csv('merged_data.csv', index=False)

print("合并完成，结果已保存到 'merged_data.csv'")

#要将数据转为numpy数组的形式，在喂给模型的的时候以这种形式。⭐
#merged_data_numpy = merged_data.values
