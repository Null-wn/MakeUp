#加载Python库
import numpy as np
#加载数据预处理模块
import pandas as pd
#加载绘图模块
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style(style="darkgrid")
#数据读取
raw_df = pd.read_csv(r"data-test.csv",encoding="utf8")
#输出前两行数据，不加参数默认输出前5行
raw_df.head(2)