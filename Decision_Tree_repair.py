import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 机器学习库
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression  # 引入逻辑回归
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix
)
from sklearn.pipeline import Pipeline

# 设置绘图风格 (解决中文乱码问题)
sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] 
plt.rcParams['axes.unicode_minus'] = False

def load_data_split(feature_file="X_final.math.one_hot.csv", target_file="y_target.math.one_hot.csv"):
    """
    加载队友处理好的分离数据集 (X 和 y 分离)
    针对用户说明：数据集是以逗号为分隔符的
    """
    print(f"正在加载特征文件: {feature_file} ...")
    print(f"正在加载标签文件: {target_file} ...")
    
    if os.path.exists(feature_file) and os.path.exists(target_file):
        try:
            # 修改点：使用逗号作为分隔符 (sep=',')
            X = pd.read_csv(feature_file, sep=',', engine='python')
            y = pd.read_csv(target_file, sep=',', engine='python')
            
            print(f"加载成功!")
            print(f"特征矩阵 X 形状: {X.shape}")
            print(f"目标变量 y 形状: {y.shape}")
            
            # 打印前几列名以验证是否正确分割
            print(f"特征列预览: {list(X.columns[:5])} ...")
            
            return X, y
        except Exception as e:
            print(f"读取文件出错: {e}")
            return None, None
    else:
        print(f"错误: 找不到文件。请确保上传了 '{feature_file}' 和 '{target_file}'。")
        return None, None

def prepare_targets(y):
    """
    检查并准备目标变量
    """
    # 将 DataFrame 转换为一维数组
    y = y.values.ravel() if hasattr(y, 'values') else y
    
    # 检查 y 的值域
    unique_vals = np.unique(y)
    print(f"目标变量中的唯一值: {unique_vals}")
    
    return y

def perform_lr_grid_search(X_train, y_train):
    """
    对 Logistic Regression 执行网格搜索 (Grid Search) 调优
    """
    print(f"\n{'='*20} 开始 Logistic Regression 模型调优 {'='*20}")
    
    # 1. 定义 Pipeline
    # 逻辑回归对特征缩放非常敏感（影响正则化），StandardScaler 是必须的
    lr_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])

    # 2. 定义超参数网格
    # C: 正则化强度的倒数，越小正则化越强（防止过拟合），越大越接近原始数据
    # solver: 优化算法，liblinear 适合小数据集，lbfgs 是默认
    param_grid = {
        'classifier__C': [0.01, 0.1, 1, 10, 100],
        'classifier__solver': ['liblinear', 'lbfgs']
    }

    print(f"搜索参数空间: {param_grid}")

    # 3. 配置 GridSearchCV
    # scoring='recall': 优先优化召回率 (捕捉更多风险学生)
    grid_search = GridSearchCV(
        estimator=lr_pipeline,
        param_grid=param_grid,
        cv=5,               # 5折交叉验证
        n_jobs=-1,          # 并行计算
        scoring='recall',   
        verbose=1
    )

    # 4. 执行搜索
    grid_search.fit(X_train, y_train)

    print(f"\n最佳参数组合: {grid_search.best_params_}")
    print(f"最佳交叉验证 Recall 分数: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def train_evaluate_model(model, X_train, X_test, y_train, y_test, model_name="Model"):
    """
    评估模型表现并绘制混淆矩阵 + 特征系数分析
    """
    print(f"\n{'='*20} 正在评估最佳模型: {model_name} {'='*20}")
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 打印核心指标
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 1. 绘制混淆矩阵
    plt.figure(figsize=(6, 4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('True Label (1=Fail/Risk)')
    plt.xlabel('Predicted Label')
    plt.show()

    # 2. 绘制特征系数 (Logistic Regression 特有功能)
    # 系数为正 -> 增加挂科风险; 系数为负 -> 降低挂科风险
    try:
        classifier = model.named_steps['classifier']
        
        # 获取系数 (对于二分类，coef_ 形状为 (1, n_features))
        coefs = classifier.coef_[0]
        
        # 按绝对值大小排序，取前 15 个最重要的特征
        indices = np.argsort(np.abs(coefs))[::-1][:15]
        
        if hasattr(X_train, 'columns'):
            feature_names = X_train.columns
            top_features = feature_names[indices]
            top_coefs = coefs[indices]
            
            # 绘图
            plt.figure(figsize=(10, 6))
            colors = ['red' if c > 0 else 'green' for c in top_coefs]
            plt.bar(range(15), top_coefs, align="center", color=colors)
            plt.xticks(range(15), top_features, rotation=45, ha='right')
            plt.axhline(0, color="black", linewidth=0.8) # 添加零线
            plt.title(f"{model_name} - Top 15 Feature Coefficients\n(Red=Increases Risk, Green=Decreases Risk)")
            plt.ylabel('Coefficient Value (Log-Odds)')
            plt.tight_layout()
            plt.show()
    except Exception as e:
        print(f"无法绘制特征系数: {e}")

def main():
    # ---------------------------------------------------------
    # 默认文件名配置 (请确保这些文件是逗号分隔的)
    feat_file = "X_final.math.one_hot.csv"
    tgt_file = "y_target.math.one_hot.csv"
    # ---------------------------------------------------------

    # 1. 加载分开的数据文件
    X, y_raw = load_data_split(feat_file, tgt_file)
    if X is None: return

    # --- [关键修复]：防止数据泄露 ---
    # 删除直接包含分数的列，强制模型使用学生背景进行预测
    leakage_cols = ['G1', 'G2', 'G3']
    cols_to_drop = [col for col in leakage_cols if col in X.columns]
    
    if cols_to_drop:
        print(f"\n[数据清洗] 检测到潜在的数据泄露列，正在删除: {cols_to_drop}")
        X = X.drop(columns=cols_to_drop)
        print(f"删除后的特征矩阵形状: {X.shape}")
    else:
        print("\n[数据清洗] 未检测到 G1/G2/G3 列，数据看起来很干净。")
    # ------------------------------------

    # 2. 准备目标变量
    y = prepare_targets(y_raw)

    # 3. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 4. 执行 Grid Search 寻找最佳模型
    best_lr_model = perform_lr_grid_search(X_train, y_train)
    
    # 5. 评估最佳模型
    train_evaluate_model(best_lr_model, X_train, X_test, y_train, y_test, model_name="Tuned Logistic Regression (Math)")

if __name__ == "__main__":
    main()