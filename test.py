import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# 设置中文字体显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def evaluate_model(model_path, data_path, task='detect', save_dir='results'):
    """评估 YOLOv8 模型并生成性能指标和可视化结果"""
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 加载模型
    model = YOLO(model_path)

    # 运行验证
    print(f"正在评估模型: {model_path}")
    results = model.val(data=data_path, task=task)

    # 保存混淆矩阵图像
    plot_confusion_matrix(results, save_dir)

    # 保存性能指标报告
    save_metrics_report(results, save_dir)

    # 如果有训练历史，绘制损失曲线
    if hasattr(model, 'history') and model.history:
        plot_training_loss(model, save_dir)

    return results


def plot_confusion_matrix(results, save_dir):
    """绘制并保存混淆矩阵"""
    # 获取混淆矩阵数据
    cm = results.confusion_matrix.matrix
    class_names = results.names.values()

    # 创建混淆矩阵图像
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.title('混淆矩阵')
    plt.tight_layout()

    # 保存图像
    cm_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"混淆矩阵已保存至: {cm_path}")


def save_metrics_report(results, save_dir):
    """保存性能指标报告"""
    # 获取指标数据
    metrics = {
        'mAP50': results.box.map50,
        'mAP50-95': results.box.map,
        'Precision': results.box.precision.mean(),
        'Recall': results.box.recall.mean(),
    }

    # 保存主要指标
    metrics_path = os.path.join(save_dir, 'metrics.csv')
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    print(f"主要指标已保存至: {metrics_path}")

    # 保存每个类别的指标
    class_metrics = {}
    for i, name in enumerate(results.names.values()):
        class_metrics[name] = {
            'Precision': results.box.p[i],
            'Recall': results.box.r[i],
            'AP50': results.box.ap50[i],
            'AP': results.box.ap_class[i]
        }

    class_metrics_path = os.path.join(save_dir, 'class_metrics.csv')
    pd.DataFrame(class_metrics).T.to_csv(class_metrics_path)
    print(f"类别指标已保存至: {class_metrics_path}")


def plot_training_loss(model, save_dir):
    """绘制训练和验证损失曲线"""
    history = model.history

    # 创建损失曲线图
    plt.figure(figsize=(12, 8))

    # 绘制Box损失
    plt.subplot(2, 2, 1)
    if 'box_loss' in history:
        plt.plot(history['box_loss'], label='训练Box损失')
    if 'val_box_loss' in history:
        plt.plot(history['val_box_loss'], label='验证Box损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title('Box损失曲线')
    plt.legend()

    # 绘制分类损失
    plt.subplot(2, 2, 2)
    if 'cls_loss' in history:
        plt.plot(history['cls_loss'], label='训练分类损失')
    if 'val_cls_loss' in history:
        plt.plot(history['val_cls_loss'], label='验证分类损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title('分类损失曲线')
    plt.legend()

    # 绘制mAP曲线
    plt.subplot(2, 2, 3)
    if 'metrics/mAP50(B)' in history:
        plt.plot(history['metrics/mAP50(B)'], label='mAP50')
    if 'metrics/mAP50-95(B)' in history:
        plt.plot(history['metrics/mAP50-95(B)'], label='mAP50-95')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.title('mAP曲线')
    plt.legend()

    plt.tight_layout()

    # 保存图像
    loss_path = os.path.join(save_dir, 'training_loss.png')
    plt.savefig(loss_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"训练损失曲线已保存至: {loss_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='YOLOv8 模型评估工具')
    parser.add_argument('--pretrained_model', type=str, default='yolov8n.pt',
                        help='预训练模型路径')
    parser.add_argument('--custom_model', type=str, default='runs/YOLO8/weights/best.pt',
                        help='自定义训练的模型路径')
    parser.add_argument('--data', type=str, required=True,
                        help='数据集配置文件路径')
    parser.add_argument('--task', type=str, default='detect',
                        choices=['detect', 'segment', 'classify', 'pose'],
                        help='任务类型')
    parser.add_argument('--save_dir', type=str, default='evaluation_results',
                        help='结果保存目录')

    args = parser.parse_args()

    # 评估预训练模型
    pretrained_save_dir = os.path.join(args.save_dir, 'pretrained')
    print(f"\n=== 评估预训练模型: {args.pretrained_model} ===")
    evaluate_model(args.pretrained_model, args.data, args.task, pretrained_save_dir)

    # 评估自定义训练的模型
    custom_save_dir = os.path.join(args.save_dir, 'custom')
    print(f"\n=== 评估自定义训练模型: {args.custom_model} ===")
    evaluate_model(args.custom_model, args.data, args.task, custom_save_dir)

    print(f"\n评估完成！结果已保存至: {args.save_dir}")


if __name__ == '__main__':
    main()    