# verify_models.py - 验证两个模型的脚本
import pickle
import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

def verify_models():
    """验证两个模型"""
    print("=" * 60)
    print("🔍 生物炭改性土SWCC预测模型验证")
    print("=" * 60)
    
    try:
        # 加载模型
        print("\n📦 正在加载模型...")
        
        with open('xgboost_optimized_results/model_group1.pkl', 'rb') as f:
            model1 = pickle.load(f)
        print("✅ 变量组一模型加载成功")
        
        with open('xgboost_optimized_results/model_group2.pkl', 'rb') as f:
            model2 = pickle.load(f)
        print("✅ 变量组二模型加载成功")
        
        # 加载特征信息
        with open('xgboost_optimized_results/feature_info.json', 'r') as f:
            feature_info = json.load(f)
        
        print(f"\n📊 变量组一特征数量: {len(feature_info['group1_features'])}")
        print(f"📊 变量组二特征数量: {len(feature_info['group2_features'])}")
        
        # 测试变量组一模型
        print("\n🧪 测试变量组一模型...")
        # 创建测试数据（需要包含所有独热编码特征）
        test_data_group1 = pd.DataFrame(columns=feature_info['group1_features'])
        
        # 创建一个测试样本
        sample1 = {
            'suction': 100.0,
            'clay': 0.2,
            'silt': 0.25,
            'sand': 0.55,
            'dd': 1.45,
            'BC': 0.05,
            'Temp': 500.0
        }
        
        # 添加独热编码特征（假设选择第一个类型）
        for i, feature in enumerate(feature_info['group1_features']):
            if feature.startswith('type_'):
                sample1[feature] = 1.0 if i == 7 else 0.0  # 假设第一个类型为1
        
        test_data_group1 = pd.DataFrame([sample1])[feature_info['group1_features']]
        
        pred1 = model1.predict(test_data_group1)[0]
        print(f"   预测值: {pred1:.4f}")
        
        # 测试变量组二模型
        print("\n🧪 测试变量组二模型...")
        sample2 = {
            'suction': 100.0,
            'clay': 0.2,
            'silt': 0.25,
            'sand': 0.55,
            'dd': 1.45,
            'BC': 0.05,
            'pH': 8.0,
            'AT': 25.0,
            'CT': 60.0
        }
        
        test_data_group2 = pd.DataFrame([sample2])[feature_info['group2_features']]
        
        pred2 = model2.predict(test_data_group2)[0]
        print(f"   预测值: {pred2:.4f}")
        
        # 测试边界条件
        print("\n🔬 测试边界条件 (BC=0)...")
        sample_boundary = {
            'suction': 100.0,
            'clay': 0.2,
            'silt': 0.25,
            'sand': 0.55,
            'dd': 1.45,
            'BC': 0.0,
            'pH': 0.0,
            'AT': 0.0,
            'CT': 0.0
        }
        
        test_boundary = pd.DataFrame([sample_boundary])[feature_info['group2_features']]
        pred_boundary = model2.predict(test_boundary)[0]
        print(f"   BC=0时预测值: {pred_boundary:.4f}")
        
        print("\n" + "=" * 60)
        print("🎉 所有模型验证完成！可以启动网页应用。")
        print("=" * 60)
        
        return True
        
    except FileNotFoundError as e:
        print(f"❌ 文件未找到: {e}")
        print("   请先运行 train_and_save_models.py 训练模型")
        return False
        
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    verify_models()