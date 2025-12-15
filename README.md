# Machine-Learning-Baseline-Models-for-teamwork
AIT201 Group Project: Student Performance Prediction

基于机器学习的学生成绩挂科风险预测系统

简介 (Project Overview)

本仓库的目标是利用机器学习算法，基于学生的家庭背景、学习习惯和社交活动等特征，提前预测学生是否存在挂科风险 (Fail Risk)。
本仓库包含五种基础模型。

这是一个典型的二分类问题 (Binary Classification)：

0 (Pass): 成绩及格 (G3 >= 10)
1 (Risk): 有挂科风险 (G3 < 10)

📂 数据集 (Dataset)

来源: 队友
科目: 数学 (Math Course)
特征处理: * One-Hot 编码 (Categorical Variables)
标准化 (Standardization)

已实现的模型 (Implemented Models)：

1.逻辑回归 (Logistic Regression)

包含 Grid Search 调优 (C, solver)。
提供特征系数 (Feature Coefficients) 可视化，分析各因素对挂科风险的正/负影响。

2.决策树 (Decision Tree)

包含 Grid Search 调优 (max_depth, min_samples_split 等)。
提供特征重要性 (Feature Importance) 可视化，展示最具影响力的特征。

3.朴素贝叶斯 (Naive Bayes - GaussianNB)

优化了 var_smoothing 参数。

4.支持向量机 (SVM)

利用 RBF 核处理高维特征边界。

5.K-近邻 (KNN)

基于距离度量的分类器，经过 Grid Search 寻找最佳 K 值。

核心亮点 (Key Features)

自动化调优: 使用 GridSearchCV 自动寻找最佳超参数。
防止作弊: 严格剔除了期中成绩 (G1, G2)，模拟真实的学期初/学期中预测场景。
深度评估: 关注 Recall (召回率) 指标，优先识别出所有潜在的风险学生，宁可误报不可漏报。


如何运行 (How to Run)

1. 环境准备

请确保安装了以下 Python 库：

pip install pandas numpy matplotlib seaborn scikit-learn


2. 数据准备

确保以下数据文件位于项目根目录：

X_final.math.one_hot.csv (特征文件)

y_target.math.one_hot.csv (标签文件)

3. 运行模型

直接运行对应的 Python 脚本即可：

# 运行逻辑回归
python "Logistic Regression.py"

# 运行决策树
python "Decision_Tree.py"

# 运行朴素贝叶斯
python "Naive Bayes_math_data.py"


📊 结果示例

运行脚本后，你将看到如下输出：

最佳参数组合 (Best Parameters)

分类报告 (Classification Report: Precision, Recall, F1-score)

混淆矩阵热力图 (Confusion Matrix Heatmap)

特征分析图 (Feature Importance/Coefficients)

Created for AIT201 Group Project, 2025.
