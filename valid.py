import os
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd
import yaml

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 配置参数
MODEL_PATHS = {
    'pretrained': 'yolov8n.pt',  # 预训练模型路径
    'custom': r'D:\py\py_pro\YOLOV8System\CarPersonDetection\models\best.pt'  # 自定义训练模型路径
}
DATA_YAML = r'D:\py\py_pro\YOLOV8System\CarPersonDetection\datasets\CarPersonData\data.yaml'  # 数据集配置文件
SAVE_DIR = 'evaluation_results'  # 结果保存目录
os.makedirs(SAVE_DIR, exist_ok=True)


def load_model(model_path):
    """加载 YOLOv8 模型"""
    return YOLO(model_path)


def evaluate_model(model, data_yaml, save_dir):
    """执行模型评估并返回结果"""
    results = model.val(
        data=data_yaml,  # 数据集配置文件
        save=True,  # 保存评估结果
        save_json=True,  # 保存 JSON 结果
        plots=False  # 不生成内置图表（后续自定义绘制）
    )
    return results


def parse_results(results, model_name, data_yaml):
    """解析评估结果"""
    # 加载类别名称
    with open(data_yaml, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
        classes = data_config['names']

    # 提取混淆矩阵
    cm = results.confusion_matrix.matrix

    # 提取性能指标
    metrics = {
        'map50': results.box.map50,
        'map50-95': results.box.map,
        'precision': results.box.precision,
        'recall': results.box.recall,
    }

    # 提取类别级指标
    class_metrics = pd.DataFrame({
        'class': classes,
        'ap50': [results.box.ap_class[i][0] for i in range(len(classes))],
        'precision': [results.box.p[i].mean().item() for i in range(len(classes))],
        'recall': [results.box.r[i].mean().item() for i in range(len(classes))]
    })

    return cm, metrics, class_metrics, classes


def plot_confusion_matrix(cm, classes, save_path):
    """绘制混淆矩阵"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('混淆矩阵')
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.savefig(save_path)
    plt.close()


def main():
    # 评估预训练模型和自定义模型
    for model_type, model_path in MODEL_PATHS.items():
        print(f"\nEvaluating {model_type.upper()} model...")
        model = load_model(model_path)
        results = evaluate_model(model, DATA_YAML, SAVE_DIR)
        cm, metrics, class_metrics, classes = parse_results(results, model_type, DATA_YAML)

        # 保存混淆矩阵
        cm_save_path = os.path.join(SAVE_DIR, f'{model_type}_confusion_matrix.png')
        plot_confusion_matrix(cm, classes, cm_save_path)

        # 输出分类报告
        class_report = classification_report(
            y_true=results.labels[:, 0].tolist(),  # 真实类别
            y_pred=[pred[0].cls.item() for pred in results.pred],  # 预测类别
            target_names=classes,
            digits=4
        )
        print(f"\n{model_type.upper()} MODEL EVALUATION REPORT:")
        print(class_report)

        # 打印性能指标
        print("\n性能指标:")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")

        # 保存类别级指标到 CSV
        class_metrics.to_csv(os.path.join(SAVE_DIR, f'{model_type}_class_metrics.csv'), index=False)


if __name__ == '__main__':
    main()