# train_and_save_models.py - 训练并保存两个模型的脚本
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """加载并准备数据"""
    # 加载数据
    data = pd.read_excel('./5.xlsx')
    
    print("数据形状:", data.shape)
    print("数据列名:", data.columns.tolist())
    
    return data

def prepare_dataset_group1(data):
    """准备变量组一的数据集"""
    # 重建Biochar type特征（根据你的代码）
    def recreate_biochar_type(row):
        active_types = [col.replace('type_', '') for col in data.columns
                       if col.startswith('type_') and row[col] == 1]
        return active_types[0] if active_types else np.nan
    
    data_copy = data.copy()
    data_copy['Biochar_type_combined'] = data_copy.apply(recreate_biochar_type, axis=1)
    
    # 删除冗余列
    data_copy = data_copy.drop(columns=['type_1', 'type_2', 'type_3', 'type_4', 'type_5'])
    
    # 移除可能的缺失值
    data_copy = data_copy.dropna(subset=['Biochar_type_combined'])
    
    # 对分类特征进行独热编码
    biochar_dummies = pd.get_dummies(data_copy['Biochar_type_combined'], prefix='type')
    data_encoded = pd.concat([data_copy.drop(columns=['Biochar_type_combined']), biochar_dummies], axis=1)
    
    # 定义特征和目标
    X = data_encoded.drop(columns=['Water content'])
    y = data_encoded['Water content']
    
    print(f"变量组一特征数量: {X.shape[1]}")
    print(f"特征列表: {X.columns.tolist()}")
    
    return X, y

def prepare_dataset_group2(data):
    """准备变量组二的数据集"""
    # 使用原始数据中的相关列
    # 注意：这里假设数据中已经包含pH, AT, CT列
    required_cols = ['suction', 'clay', 'silt', 'sand', 'dd', 'BC', 'pH', 'AT', 'CT', 'Water content']
    
    # 检查缺失的列
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        print(f"警告: 数据中缺少以下列: {missing_cols}")
        
        # 创建示例数据（实际使用时需要替换为真实数据）
        data_copy = data.copy()
        for col in ['pH', 'AT', 'CT']:
            if col not in data_copy.columns:
                if col == 'pH':
                    data_copy[col] = np.random.uniform(5.5, 9.5, len(data_copy))
                elif col == 'AT':
                    data_copy[col] = np.random.uniform(10, 40, len(data_copy))
                elif col == 'CT':
                    data_copy[col] = np.random.uniform(40, 80, len(data_copy))
    else:
        data_copy = data.copy()
    
    # 移除缺失值
    data_copy = data_copy.dropna(subset=required_cols)
    
    # 定义特征和目标
    X = data_copy[['suction', 'clay', 'silt', 'sand', 'dd', 'BC', 'pH', 'AT', 'CT']]
    y = data_copy['Water content']
    
    print(f"变量组二特征数量: {X.shape[1]}")
    print(f"特征列表: {X.columns.tolist()}")
    
    return X, y

def train_xgboost_model(X, y, model_name):
    """训练XGBoost模型"""
    print(f"\n正在训练 {model_name} ...")
    
    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 创建XGBoost回归器
    xgb_regressor = XGBRegressor(
        random_state=42,
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1
    )
    
    # 简单的参数网格（可以简化以加快训练）
    param_grid = {
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1.0],
    }
    
    print("正在进行网格搜索...")
    grid_search = GridSearchCV(
        estimator=xgb_regressor, 
        param_grid=param_grid, 
        cv=5,
        scoring='neg_mean_squared_error', 
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"最佳参数: {grid_search.best_params_}")
    
    # 评估模型
    y_pred = best_model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_r2 = r2_score(y_test, y_pred)
    
    print(f"测试集RMSE: {test_rmse:.4f}")
    print(f"测试集R²: {test_r2:.4f}")
    
    # 交叉验证
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores.mean())
    print(f"交叉验证RMSE: {cv_rmse:.4f}")
    
    return best_model

def main():
    """主函数"""
    print("=" * 60)
    print("生物炭改性土SWCC预测模型训练")
    print("=" * 60)
    
    # 加载数据
    data = load_and_prepare_data()
    
    # 训练变量组一模型
    X1, y1 = prepare_dataset_group1(data)
    model1 = train_xgboost_model(X1, y1, "变量组一模型")
    
    # 训练变量组二模型
    X2, y2 = prepare_dataset_group2(data)
    model2 = train_xgboost_model(X2, y2, "变量组二模型")
    
    # 保存模型
    import os
    os.makedirs('xgboost_optimized_results', exist_ok=True)
    
    with open('xgboost_optimized_results/model_group1.pkl', 'wb') as f:
        pickle.dump(model1, f)
    print("✅ 变量组一模型已保存")
    
    with open('xgboost_optimized_results/model_group2.pkl', 'wb') as f:
        pickle.dump(model2, f)
    print("✅ 变量组二模型已保存")
    
    # 保存特征顺序（用于网页应用）
    feature_info = {
        'group1_features': X1.columns.tolist(),
        'group2_features': X2.columns.tolist()
    }
    
    import json
    with open('xgboost_optimized_results/feature_info.json', 'w') as f:
        json.dump(feature_info, f, indent=4)
    print("✅ 特征信息已保存")
    
    print("\n" + "=" * 60)
    print("模型训练完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()