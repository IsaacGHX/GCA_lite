import numpy as np
import torch
import torch.nn as nn
import copy
from .evaluate_visualization import *
import torch.optim.lr_scheduler as lr_scheduler
import time
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_


def train_multi_gan(generators, discriminators, dataloaders,
                    window_sizes,
                    y_scaler, train_xes, train_y, val_xes, val_y,
                    distill,
                    num_epochs,
                    output_dir,
                    device,
                    init_GDweight=[
                        [1, 0, 0, 1.0],  # alphas_init
                        [0, 1, 0, 1.0],  # betas_init
                        [0., 0, 1, 1.0]  # gammas_init...
                    ],
                    final_GDweight=[
                        [0.333, 0.333, 0.333, 1.0],  # alphas_final
                        [0.333, 0.333, 0.333, 1.0],  # betas_final
                        [0.333, 0.333, 0.333, 1.0]  # gammas_final...
                    ]
                    ):
    N = len(generators)

    assert N == len(discriminators)
    assert N == len(window_sizes)
    assert N > 1

    g_learning_rate = 2e-5
    d_learning_rate = 2e-5

    # 二元交叉熵【损失函数，可能会有问题
    criterion = nn.BCELoss()

    optimizers_G = [torch.optim.AdamW(model.parameters(), lr=g_learning_rate, betas=(0.9, 0.999))
                    for model in generators]

    # 为每个优化器设置 ReduceLROnPlateau 调度器
    schedulers = [lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=16, min_lr=1e-7)
                  for optimizer in optimizers_G]

    optimizers_D = [torch.optim.Adam(model.parameters(), lr=d_learning_rate, betas=(0.9, 0.999))
                    for model in discriminators]

    best_epoch = [-1 for _ in range(N)]  #

    # 定义生成历史记录的关键字
    """
    以三个为例，keys长得是这样得的：
    ['G1', 'G2', 'G3', 
    'D1', 'D2', 'D3', 
    'MSE_G1', 'MSE_G2', 'MSE_G3', 
    'val_G1', 'val_G2', 'val_G3', 
    'D1_G1', 'D2_G1', 'D3_G1', 'D1_G2', 'D2_G2', 'D3_G2', 'D1_G3', 'D2_G3', 'D3_G3'
    ]
    """

    keys = []
    g_keys = [f'G{i}' for i in range(1, N + 1)]
    d_keys = [f'D{i}' for i in range(1, N + 1)]
    MSE_g_keys = [f'MSE_G{i}' for i in range(1, N + 1)]
    val_loss_keys = [f'val_G{i}' for i in range(1, N + 1)]

    keys.extend(g_keys)
    keys.extend(d_keys)
    keys.extend(MSE_g_keys)
    keys.extend(val_loss_keys)

    d_g_keys = []
    for g_key in g_keys:
        for d_key in d_keys:
            d_g_keys.append(d_key + "_" + g_key)
    keys.extend(d_g_keys)

    # 创建包含每个值为np.zeros(num_epochs)的字典
    hists_dict = {key: np.zeros(num_epochs) for key in keys}

    best_mse = [float('inf') for _ in range(N)]

    best_model_state = [None for _ in range(N)]

    patience_counter = 0
    patience = 50
    feature_num = train_xes[0].shape[2]
    target_num = train_y.shape[-1]

    print("start training")
    for epoch in range(num_epochs):
        epo_start = time.time()

        if epoch < 10:
            weight_matrix = torch.tensor(init_GDweight).to(device)
        else:
            weight_matrix = torch.tensor(final_GDweight).to(device)

        keys = []
        keys.extend(g_keys)
        keys.extend(d_keys)
        keys.extend(MSE_g_keys)
        keys.extend(d_g_keys)

        loss_dict = {key: [] for key in keys}

        # use the gap the equalize the length of different generators
        gaps = [window_sizes[-1] - window_sizes[i] for i in range(N - 1)]

        for batch_idx, (x_last, y_last) in enumerate(dataloaders[-1]):
            # TODO: maybe try to random select a gap from the whole time windows
            x_last = x_last.to(device)
            y_last = y_last.to(device)

            X = []
            Y = []

            for gap in gaps:
                X.append(x_last[:, gap:, :])
                Y.append(y_last[:, gap:, :])
            X.append(x_last.to(device))
            Y.append(y_last.to(device))

            for i in range(N):
                generators[i].eval()
                discriminators[i].train()

            loss_D, lossD_G = discriminate_fake(X, Y,
                                                generators, discriminators,
                                                window_sizes, target_num,
                                                criterion, weight_matrix,
                                                device, mode="train_D")

            # 3. 存入 loss_dict
            for i in range(N):
                loss_dict[d_keys[i]].append(loss_D[i].item())

            for i in range(1, N + 1):
                for j in range(1, N + 1):
                    key = f'D{i}_G{j}'
                    loss_dict[key].append(lossD_G[i - 1, j - 1].item())

            # 根据批次的奇偶性交叉训练两个GAN
            # if batch_idx% 2 == 0:
            for optimizer_D in optimizers_D:
                optimizer_D.zero_grad()

            # TODO: to see whether there is need to add together

            # for i, loss in enumerate(loss_D):
            #     if i != N - 1:
            #         loss.backward(retain_graph=True)
            #     else:
            #         loss.backward()

            loss_D.sum(dim=0).backward()

            for i in range(N):
                optimizers_D[i].step()
                discriminators[i].eval()
                generators[i].train()

            '''训练生成器'''
            loss_G, loss_mse_G = discriminate_fake(X, Y,
                                                   generators, discriminators,
                                                   window_sizes, target_num,
                                                   criterion, weight_matrix,
                                                   device,
                                                   mode="train_G")

            for i in range(N):
                loss_dict[g_keys[i]].append(loss_G[i].item())
                loss_dict["MSE_" + g_keys[i]].append(loss_mse_G[i].item())

            for optimizer_G in optimizers_G:
                optimizer_G.zero_grad()

            # for i, loss in enumerate(loss_G):
                # if i != N - 1:
                #     loss.backward(retain_graph=True)
                # else:
                #     loss.backward()
            loss_G.sum(dim=0).backward()

            for optimizer_G in optimizers_G:
                optimizer_G.step()

        for key in loss_dict.keys():
            hists_dict[key][epoch] = np.mean(loss_dict[key])

        improved = [False] * 3
        for i in range(N):

            hists_dict[val_loss_keys[i]][epoch] = validate(generators[i], val_xes[i], val_y[i])

            if hists_dict[val_loss_keys[i]][epoch].item() < best_mse[i]:
                best_mse[i] = hists_dict[val_loss_keys[i]][epoch]
                best_model_state[i] = copy.deepcopy(generators[i].state_dict())
                best_epoch[i] = epoch + 1
                improved[i] = True

            schedulers[i].step(hists_dict[val_loss_keys[i]][epoch])

        if distill and epoch % 10 == 0:
            losses = [hists_dict[val_loss_keys[i]][epoch] for i in range(N)]
            rank = np.argsort(losses)
            do_distill(rank, generators, dataloaders, optimizers_G, window_sizes, device)
        if epoch % 10 == 0:
            G_losses = [hists_dict[val_loss_keys[i]][epoch] for i in range(N)]
            D_losses = [np.mean(loss_dict[d_keys[i]]) for i in range(N)]
            G_rank = np.argsort(G_losses)
            D_rank = np.argsort(D_losses)

            refine_best_models_with_real_data_v2(
                G_rank, D_rank,
                generators=generators,
                discriminators=discriminators,
                g_optimizers=optimizers_G,
                d_optimizers=optimizers_D,
                dataloaders=dataloaders,
                window_sizes=window_sizes,
                device_G="cuda:0",  # or "cuda:1", assign as needed
                device_D="cuda:0"
            )

        # 每个epoch结束时，打印训练过程中的损失
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        # 动态生成打印字符串
        log_str = ", ".join(
            f"G{i + 1}: {hists_dict[key][epoch]:.8f}"
            for i, key in enumerate(val_loss_keys)
        )
        print(f"Validation MSE {log_str}")
        print(f"patience counter:{patience_counter}")
        if not any(improved):
            patience_counter += 1
        else:
            patience_counter = 0

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

        epo_end = time.time()
        print(f"Epoch time: {epo_end - epo_start:.4f}")

    data_G = [[[] for _ in range(4)] for _ in range(N)]
    data_D = [[[] for _ in range(4)] for _ in range(N)]

    for i in range(N):
        for j in range(N + 1):
            if j < N:
                data_G[i][j] = hists_dict[f"D{j+1}_G{i+1}"]
                data_D[i][j] = hists_dict[f"D{i+1}_G{j+1}"]
            elif j == N:
                data_G[i][j] = hists_dict[g_keys[i]]
                data_D[i][j] = hists_dict[d_keys[i]]

    plot_generator_losses(data_G, output_dir)
    plot_discriminator_losses(data_D, output_dir)

    # overall G&D
    visualize_overall_loss(data_G[-1][:][:epoch], data_D[-1][:][:epoch], output_dir)

    hist_MSE_G = [[] for _ in range(N)]
    hist_val_loss = [[] for _ in range(N)]
    for i in range(N):
        hist_MSE_G[i] = hists_dict[f"MSE_G{i+1}"][:epoch]
        hist_val_loss[i] = hists_dict[f"val_G{i+1}"][:epoch]

    plot_mse_loss(hist_MSE_G, hist_val_loss, epoch, output_dir)

    for i in range(N):
        print(f"G{i + 1} best epoch: ", best_epoch[i])

    results = evaluate_best_models(generators, best_model_state, train_xes, train_y, val_xes, val_y, y_scaler, output_dir)
    return results


def discriminate_fake(X, Y,
                      generators, discriminators,
                      window_sizes, target_num,
                      criterion, weight_matrix,
                      device,
                      mode):
    assert mode in ["train_D", "train_G"]

    N = len(generators)

    # discriminator output for real data
    dis_real_outputs = [model(y) for (model, y) in zip(discriminators, Y)]
    real_labels = [torch.ones_like(dis_real_output).to(device) for dis_real_output in dis_real_outputs]
    fake_data_G = [generator(x) for (generator, x) in zip(generators, X)]  # cannot be omitted

    # 判别器对真实数据损失
    lossD_real = [criterion(dis_real_output, real_label) for (dis_real_output, real_label) in
                  zip(dis_real_outputs, real_labels)]

    if mode == "train_D":
        # G1生成的数据
        fake_data_temp_G = [fake_data.detach() for fake_data in fake_data_G]
        # 拼接之后可以让生成的假数据，既包含假数据又包含真数据，
        fake_data_temp_G = [torch.cat([y[:, :window_size, :], fake_data.reshape(-1, 1, target_num)], axis=1)
                            for (y, window_size, fake_data) in zip(Y, window_sizes, fake_data_temp_G)]
    elif mode == "train_G":
        # 拼接之后可以让生成的假数据，既包含假数据又包含真数据，
        fake_data_temp_G = [torch.cat([y[:, :window_size, :], fake_data.reshape(-1, 1, target_num)], axis=1)
                            for (y, window_size, fake_data) in zip(Y, window_sizes, fake_data_G)]

    # 判别器对伪造数据损失
    # 三个生成器的结果的数据对齐
    fake_data_GtoD = {}
    for i in range(N):
        for j in range(N):
            if i < j:
                fake_data_GtoD[f"G{i + 1}ToD{j + 1}"] = torch.cat(
                    [Y[j][:, :window_sizes[j] - window_sizes[i], :], fake_data_temp_G[i]], axis=1)
            elif i > j:
                fake_data_GtoD[f"G{i + 1}ToD{j + 1}"] = fake_data_temp_G[i][:, window_sizes[i] - window_sizes[j]:, :]
            elif i == j:
                fake_data_GtoD[f"G{i + 1}ToD{j + 1}"] = fake_data_temp_G[i]

    fake_labels = [torch.zeros_like(real_label).to(device) for real_label in real_labels]

    dis_fake_outputD = []
    for i in range(N):
        row = []
        for j in range(N):
            out = discriminators[i](fake_data_GtoD[f"G{j + 1}ToD{i + 1}"])
            row.append(out)
        if mode == "train_D":
            row.append(lossD_real[i])
        dis_fake_outputD.append(row)  # dis_fake_outputD[i][j] = Di(Gj)

    if mode == "train_D":
        loss_matrix = torch.zeros(N, N + 1, device=device)  # device 取决于你的模型位置
        weight = weight_matrix  # [N, N+1]
        for i in range(N):
            for j in range(N + 1):
                if j < N:
                    loss_matrix[i, j] = criterion(dis_fake_outputD[i][j], fake_labels[i])
                elif j == N:
                    loss_matrix[i, j] = dis_fake_outputD[i][j]
    elif mode == "train_G":
        loss_matrix = torch.zeros(N, N, device=device)  # device 取决于你的模型位置
        weight = weight_matrix[:, :-1].clone().detach()  # [N, N]
        for i in range(N):
            for j in range(N):
                loss_matrix[i, j] = criterion(dis_fake_outputD[i][j], real_labels[i])

    loss_DorG = torch.multiply(weight, loss_matrix).sum(dim=1)  # [N, N] --> [N, ]

    if mode == "train_G":
        loss_mse_G = [F.mse_loss(fake_data.squeeze(), y[:, -1, :].squeeze()) for (fake_data, y) in zip(fake_data_G, Y)]
        loss_matrix = loss_mse_G
        loss_DorG = loss_DorG + torch.stack(loss_matrix).to(device)

    return loss_DorG, loss_matrix


def do_distill(rank, generators, dataloaders, optimizers, window_sizes, device,
               *,
               alpha: float = 0.7,  # 软目标权重
               temperature: float = 2.0,  # 温度系数
               grad_clip: float = 1.0  # 梯度裁剪上限 (L2‑norm)
               ):
    teacher_generator = generators[rank[0]]  # Teacher generator is ranked first
    student_generator = generators[rank[-1]]  # Student generator is ranked last
    student_optimizer = optimizers[rank[-1]]
    teacher_generator.eval()
    student_generator.train()
    # term of teacher is longer
    if window_sizes[rank[0]] > window_sizes[rank[-1]]:
        distill_dataloader = dataloaders[rank[0]]
    else:
        distill_dataloader = dataloaders[rank[-1]]
    gap = window_sizes[rank[0]] - window_sizes[rank[-1]]
    # Distillation process: Teacher generator to Student generator
    for batch_idx, (x, y) in enumerate(distill_dataloader):

        y = y[:, -1, :]
        y = y.to(device)
        if gap > 0:
            x_teacher = x
            x_student = x[:, gap:, :]
        else:
            x_teacher = x[:, (-1) * gap:, :]
            x_student = x
        x_teacher = x_teacher.to(device)
        x_student = x_student.to(device)

        # Forward pass with teacher generator
        teacher_output = teacher_generator(x_teacher).detach()/ temperature

        # Forward pass with student generator
        student_output = student_generator(x_student)/ temperature

        # Calculate distillation loss (MSE between teacher and student generator's outputs)
        soft_loss = F.mse_loss(student_output, teacher_output) * (alpha * temperature ** 2)
        hard_loss = F.mse_loss(student_output * temperature, y) * (1 - alpha)
        distillation_loss = soft_loss + hard_loss

        # Backpropagate the loss and update student generator
        student_optimizer.zero_grad()
        distillation_loss.backward()

        if grad_clip is not None:
            clip_grad_norm_(student_generator.parameters(), grad_clip)

        student_optimizer.step()  # Assuming same optimizer for all generators, modify as needed


def refine_best_models_with_real_data_v2(
        G_rank, D_rank, generators, discriminators, g_optimizers, d_optimizers,
        dataloaders, window_sizes, device_G="cuda:0", device_D="cuda:0"
):
    best_G_idx = G_rank[0]
    best_D_idx = D_rank[0]

    generator = generators[best_G_idx]
    discriminator = discriminators[best_D_idx]
    g_optimizer = g_optimizers[best_G_idx]
    d_optimizer = d_optimizers[best_D_idx]
    dataloader_G = dataloaders[best_G_idx]
    dataloader_D = dataloaders[best_D_idx]
    window_size_D = window_sizes[best_D_idx]

    def train_generator():
        generator.to(device_G)
        generator.train()
        for x, y in dataloader_G:
            x = x.to(device_G)
            y = y[:, -1, :].to(device_G)  # 用最后一个时间步
            g_optimizer.zero_grad()
            pred = generator(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
            g_optimizer.step()

    def train_discriminator():
        discriminator.to(device_D)
        discriminator.train()
        for _, y in dataloader_D:
            y_real = y[:, -window_size_D:, :].to(device_D)
            y_fake = y_real + torch.randn_like(y_real) * 0.05
            label_real = torch.ones((y_real.size(0), 1)).to(device_D)
            label_fake = torch.zeros((y_fake.size(0), 1)).to(device_D)

            d_optimizer.zero_grad()
            out_real = discriminator(y_real)
            out_fake = discriminator(y_fake)
            loss_real = F.binary_cross_entropy(out_real, label_real)
            loss_fake = F.binary_cross_entropy(out_fake, label_fake)
            loss = loss_real + loss_fake
            loss.backward()
            d_optimizer.step()
