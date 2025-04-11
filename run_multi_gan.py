import argparse
from time_series_gca import GCA_time_series
import pandas as pd
import os


def run_experiments(args):
    # 创建保存结果的CSV文件
    results_file = os.path.join(args.output_dir, "gca_GT_NPDC_market.csv")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print("Output directory created")

    gca = GCA_time_series(args.N_pairs, args.batch_size, args.num_epochs,
                          args.generators, args.discriminators,
                          args.ckpt_dir, args.output_dir,
                          args.window_sizes,
                          initial_learning_rate=args.lr,
                          train_split=args.train_split,
                          do_distill= args.distill,
                          device=args.device,
                          seed=args.random_seed)

    # feature_columns = list(range(2,56))
    feature_columns = []
    # feature_columns = list(range(35,36))
    # target_columns = [[i] for i in range(1, 22)]
    target_columns = [list(range(1, 2))]

    for target in target_columns:
        # for target,feature in zip(target_columns,feature_columns):
        # 运行实验，获取结果
        target_feature_columns = feature_columns
        # target_feature_columns = feature_columns
        # target_feature_columns=target_feature_columns.extend(target)
        target_feature_columns.extend(target)
        # target_feature_columns.append(target)
        print("using features:", target_feature_columns)

        gca.process_data(args.data_path, target, target_feature_columns)
        gca.init_dataloader()
        gca.init_model()
        results = gca.train()

        # 将结果保存到CSV
        result_row = {
            "feature_columns": feature_columns,
            "target_columns": target,
            "train_mse": results["train_mse"],
            "train_mae": results["train_mae"],
            "train_rmse": results["train_rmse"],
            "train_mape": results["train_mape"],
            "train_mse_per_target": results["train_mse_per_target"],
            "test_mse": results["test_mse"],
            "test_mae": results["test_mae"],
            "test_rmse": results["test_rmse"],
            "test_mape": results["test_mape"],
            "test_mse_per_target": results["test_mse_per_target"]
        }
        df = pd.DataFrame([result_row])
        df.to_csv(results_file, mode='a', header=not pd.io.common.file_exists(results_file), index=False)


if __name__ == "__main__":
    # 使用argparse解析命令行参数
    parser = argparse.ArgumentParser(description="Run experiments for triple GAN model")
    parser.add_argument('--data_path', type=str, required=False, help="Path to the input data file",
                        default="database/cleaned_data.csv")
    parser.add_argument('--output_dir', type=str, required=False, help="Directory to save the output",
                        default="out_put/multi")
    parser.add_argument('--ckpt_dir', type=str, required=False, help="Directory to save the checkpoints",
                        default="ckpt")
    parser.add_argument('--window_sizes', type=list, help="Window size for first dimension", default=[5, 10, 15])
    parser.add_argument('--N_pairs', "-n", type=int, help="Window size for first dimension", default=3)
    parser.add_argument('--generators', "-gens", type=list, help="Window size for first dimension",
                        default=["gru", "lstm", "transformer"])
    parser.add_argument('--discriminators', "-discs", type=list, help="Window size for first dimension", default=None)
    parser.add_argument('--distill', type=bool, help="Whether to do distillation", default=True)
    parser.add_argument('--device', type=list, help="Device sets", default=[0])

    parser.add_argument('--num_epochs', type=int, help="epoch", default=10000)
    parser.add_argument('--lr', type=int, help="initial learning rate", default=2e-4)
    parser.add_argument('--batch_size', type=int, help="Batch size for training", default=64)
    parser.add_argument('--train_split', type=float, help="Train-test split ratio", default=0.8)
    parser.add_argument('--random_seed', type=int, help="Random seed for reproducibility", default=3407)

    args = parser.parse_args()

    print("===== Running with the following arguments =====")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("===============================================")

    # 调用run_experiments函数
    run_experiments(args)
