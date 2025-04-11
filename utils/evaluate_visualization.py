import torch.nn.functional as F
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import torch
import matplotlib.pyplot as plt


def validate(model, val_x, val_y):
    model.eval()  # 将模型设置为评估模式
    with torch.no_grad():  # 禁止计算梯度
        val_x = val_x.clone().detach().float()

        # 检查val_y的类型，如果是numpy.ndarray则转换为torch.Tensor
        if isinstance(val_y, np.ndarray):
            val_y = torch.tensor(val_y).float()
        else:
            val_y = val_y.clone().detach().float()

        # 使用模型进行预测
        predictions = model(val_x).cpu().numpy()
        val_y = val_y.cpu().numpy()

        # 计算均方误差（MSE）作为验证损失
        mse_loss = F.mse_loss(torch.tensor(predictions).float().squeeze(), torch.tensor(val_y).float().squeeze())
        return mse_loss


def plot_generator_losses(data_G, output_dir):
    """
    绘制 G1、G2、G3 的损失曲线。

    Args:
        data_G1 (list): G1 的损失数据列表，包含 [histD1_G1, histD2_G1, histD3_G1, histG1]。
        data_G2 (list): G2 的损失数据列表，包含 [histD1_G2, histD2_G2, histD3_G2, histG2]。
        data_G3 (list): G3 的损失数据列表，包含 [histD1_G3, histD2_G3, histD3_G3, histG3]。
    """

    all_data = data_G
    N = len(all_data)

    plt.figure(figsize=(5 * N, 5))  # 可选：设置图形大小

    # 循环绘制 G1、G2、G3 的损失曲线
    for i, data in enumerate(all_data):
        plt.subplot(1, N, i + 1)  # 创建子图
        for j, acc in enumerate(data):
            plt.plot(data[0], label=f"G{i + 1} against D{j} Loss")
            if j == N:
                plt.plot(data[j], label=f"combined G{i + 1} Loss")

        plt.xlabel("Epoch")
        plt.ylabel(f"G{i + 1} Loss")
        plt.title(f"G{i + 1} Loss over Epochs")
        plt.legend()

    # 如果需要显示整个图形，可以添加 plt.show()
    plt.savefig(os.path.join(output_dir, "generator_losses.png"))


def plot_discriminator_losses(data_D, output_dir):
    all_data = data_D

    N = len(all_data)

    plt.figure(figsize=(5 * N, 5))  # 可选：设置图形大小

    # 循环绘制 G1、G2、G3 的损失曲线
    for i, data in enumerate(all_data):
        plt.subplot(1, N, i + 1)  # 创建子图
        for j, acc in enumerate(data):
            plt.plot(data[0], label=f"D{i + 1} against G{j} Loss")
            if j == N:
                plt.plot(data[j], label=f"combined D{i + 1} Loss")

        plt.xlabel("Epoch")
        plt.ylabel(f"D{i + 1} Loss")
        plt.title(f"D{i + 1} Loss over Epochs")
        plt.legend()

    # 如果需要显示整个图形，可以添加 plt.show()
    plt.savefig(os.path.join(output_dir, "discriminator_losses.png"))


def visualize_overall_loss(histG, histD, output_dir):
    N = len(histG)
    plt.figure(figsize=(4*N, 2*N))  # 可选：设置图形大小
    for i, (histG, histD) in enumerate(zip(histG, histD)):
        plt.plot(histG, label=f"G{i} Loss")
        plt.plot(histD, label=f"D{i} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Generator and Discriminator Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "overall_losses.png"))


def plot_mse_loss(hist_MSE_G, hist_val_loss, num_epochs,
                  output_dir):
    """
    绘制训练过程中和验证集上的MSE损失变化曲线

    参数：
    hist_MSE_G1, hist_MSE_G2, hist_MSE_G3 : 训练过程中各生成器的MSE损失
    hist_val_loss1, hist_val_loss2, hist_val_loss3 : 验证集上各生成器的MSE损失
    num_epochs : 训练的epoch数
    """
    N = len(hist_MSE_G)
    plt.figure(figsize=(4 * N, 2 * N))  # 可选：设置图形大小
    for i, (MSE, val_loss) in enumerate(zip(hist_MSE_G, hist_val_loss)):
        # 绘制训练集MSE损失曲线
        plt.plot(range(num_epochs), MSE, label=f"Train MSE G{i+1}")
        # 绘制验证集MSE损失曲线
        plt.plot(range(num_epochs), val_loss, label=f"Val MSE G{i+1}", alpha=0.5)

    plt.title("MSE Loss for Generators (Train and Validation)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()

    plt.tight_layout()  # 自动调整子图间距
    plt.savefig(os.path.join(output_dir, "mse_losses.png"))


def inverse_transform(predictions, scaler):
    """ 使用y_scaler逆转换预测结果 """
    return scaler.inverse_transform(predictions)


def compute_metrics(true_values, predicted_values):
    """计算MSE, MAE, RMSE, MAPE"""
    mse = mean_squared_error(true_values, predicted_values)
    mae = mean_absolute_error(true_values, predicted_values)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((true_values - predicted_values) / true_values)) * 100
    per_target_mse = np.mean((true_values - predicted_values) ** 2, axis=0)  # 新增
    return mse, mae, rmse, mape, per_target_mse


def plot_fitting_curve(true_values, predicted_values, output_dir, model_name):
    """绘制拟合曲线并保存结果"""
    plt.figure(figsize=(10, 6))
    plt.plot(true_values, label='True Values')
    plt.plot(predicted_values, label='Predicted Values')
    plt.title(f'{model_name} Fitting Curve')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(f'{output_dir}/{model_name}_fitting_curve.png')
    plt.close()


def save_metrics(metrics, output_dir, model_name):
    """保存MSE, MAE, RMSE, MAPE到文件"""
    with open(f'{output_dir}/{model_name}_metrics.txt', 'w') as f:
        f.write("MSE: {}\n".format(metrics[0]))
        f.write("MAE: {}\n".format(metrics[1]))
        f.write("RMSE: {}\n".format(metrics[2]))
        f.write("MAPE: {}\n".format(metrics[3]))


def evaluate_best_models(generators, best_model_state, train_xes, train_y, test_xes, test_y, y_scaler, output_dir):
    N = len(generators)

    # 加载模型并设为 eval
    for i in range(N):
        generators[i].load_state_dict(best_model_state[i])
        generators[i].eval()

    train_y_inv = inverse_transform(train_y, y_scaler)
    test_y_inv = inverse_transform(test_y, y_scaler)

    train_preds_inv = []
    test_preds_inv = []
    train_metrics_list = []
    test_metrics_list = []

    with torch.no_grad():
        for i in range(N):
            train_pred = generators[i](train_xes[i]).cpu().numpy()
            train_pred_inv = inverse_transform(train_pred, y_scaler)
            train_preds_inv.append(train_pred_inv)
            train_metrics = compute_metrics(train_y_inv, train_pred_inv)
            train_metrics_list.append(train_metrics)
            plot_fitting_curve(train_y_inv, train_pred_inv, output_dir, f'G{i+1}_Train')
            print(f"Train Metrics for G{i+1}: MSE={train_metrics[0]:.4f}, MAE={train_metrics[1]:.4f}, RMSE={train_metrics[2]:.4f}, MAPE={train_metrics[3]:.4f}")

        for i in range(N):
            test_pred = generators[i](test_xes[i]).cpu().numpy()
            test_pred_inv = inverse_transform(test_pred, y_scaler)
            test_preds_inv.append(test_pred_inv)
            test_metrics = compute_metrics(test_y_inv, test_pred_inv)
            test_metrics_list.append(test_metrics)
            plot_fitting_curve(test_y_inv, test_pred_inv, output_dir, f'G{i+1}_Test')
            print(f"Test Metrics for G{i+1}: MSE={test_metrics[0]:.4f}, MAE={test_metrics[1]:.4f}, RMSE={test_metrics[2]:.4f}, MAPE={test_metrics[3]:.4f}")

    # 构造返回结果
    result = {
        "train_mse":  [m[0] for m in train_metrics_list],
        "train_mae":  [m[1] for m in train_metrics_list],
        "train_rmse": [m[2] for m in train_metrics_list],
        "train_mape": [m[3] for m in train_metrics_list],
        "train_mse_per_target": [m[4] for m in train_metrics_list],

        "test_mse":  [m[0] for m in test_metrics_list],
        "test_mae":  [m[1] for m in test_metrics_list],
        "test_rmse": [m[2] for m in test_metrics_list],
        "test_mape": [m[3] for m in test_metrics_list],
        "test_mse_per_target": [m[4] for m in test_metrics_list],
    }

    return result

