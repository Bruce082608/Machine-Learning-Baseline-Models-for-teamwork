import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC                           #SVM
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix
)
from sklearn.pipeline import Pipeline

sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] # Mac上防止中文乱码，Windows可改为 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

def load_data_split(feature_file="X_final.math.one_hot.csv", target_file="y_target.math.one_hot.csv"):
    print(f"正在加载特征文件: {feature_file} ...")
    print(f"正在加载标签文件: {target_file} ...")
    
    if os.path.exists(feature_file) and os.path.exists(target_file):
        try:
            X = pd.read_csv(feature_file, sep=',', engine='python')
            y = pd.read_csv(target_file, sep=',', engine='python')
            
            print(f"加载成功!")
            print(f"特征矩阵 X 形状: {X.shape}")
            print(f"目标变量 y 形状: {y.shape}")
            
            print(f"特征列示例: {list(X.columns[:5])} ...")
            
            return X, y
        except Exception as e:
            print(f"读取文件出错: {e}")
            return None, None
    else:
        print(f"错误: 找不到文件。请确保上传了 '{feature_file}' 和 '{target_file}'。")
        return None, None

def prepare_targets(y):
    y = y.values.ravel() if hasattr(y, 'values') else y
    
    unique_vals = np.unique(y)
    print(f"目标变量中的唯一值: {unique_vals}")
    
    return y

def train_evaluate_model(model, X_train, X_test, y_train, y_test, model_name="Model"):
    print(f"\n{'='*20} 正在评估: {model_name} {'='*20}")
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    plt.figure(figsize=(6, 4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('True Label (1=Fail/Risk)')
    plt.xlabel('Predicted Label')
    plt.show()

    return model

def main():
    feat_file = "X_final.math.one_hot.csv"
    tgt_file = "y_target.math.one_hot.csv"

    X, y_raw = load_data_split(feat_file, tgt_file)
    if X is None: return

    y = prepare_targets(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    svm_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('classifier', SVC(random_state=42)) 
    ])
    
    train_evaluate_model(svm_pipeline, X_train, X_test, y_train, y_test, model_name="Support Vector Machine (SVM)")

if __name__ == "__main__":
    main()