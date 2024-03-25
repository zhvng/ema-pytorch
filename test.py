import math
import os
import shutil
from matplotlib import pyplot as plt
import numpy as np
import torch
from ema_pytorch import PostHocEMA
from ema_pytorch.post_hoc_ema import sigma_rel_to_gamma


N = 5000
t = np.arange(1, N + 1)

def train(gammas, idx, update_every = 2):
    torch.manual_seed(0)
    np.random.seed(0)
    # your neural network as a pytorch module

    net = torch.nn.Linear(1, 1)

    brownian_motion = (
        10 + math.sqrt(N) * np.sin(t / 300) + np.cumsum(np.random.normal(0, 0.5, N))
    )

    # wrap your neural network, specify the sigma_rels or gammas

    outdir = f'./post-hoc-ema-checkpoints_{idx}'
    if os.path.exists(outdir):
        shutil.rmtree(outdir)

    emas = PostHocEMA(
        net,
        gammas = gammas,           # a tuple with the hyperparameter for the multiple EMAs. you need at least 2 here to synthesize a new one
        update_every = update_every,                  # how often to actually update, to save on compute (updates every 10th .update() call)
        checkpoint_every_num_steps =  500,
        checkpoint_folder = outdir  # the folder of saved checkpoints for each sigma_rel (gamma) across timesteps with the hparam above, used to synthesizing a new EMA model after training
    )


    net.train()

    gt_weight = []
    ema_low_weight = []
    ema_high_weight = []
    for i in range(N):
        # mutate your network, with SGD or otherwise

        with torch.no_grad():
            net.weight.copy_(torch.tensor(brownian_motion[i]).float())

        # you will call the update function on your moving average wrapper

        emas.update()

        gt_weight.append(net.weight.item())
        ema_low_weight.append(emas.ema_models[0].ema_model.weight.item())
        ema_high_weight.append(emas.ema_models[1].ema_model.weight.item())

    # now that you have a few checkpoints
    # you can synthesize an EMA model with a different sigma_rel (say 0.15)

    return emas, gt_weight, ema_low_weight, ema_high_weight

low_sigma_rel = 0.05
high_sigma_rel = 0.1
target_sigma_rel = 0.15


low_gamma_1 = sigma_rel_to_gamma(low_sigma_rel)
high_gamma_1 = sigma_rel_to_gamma(high_sigma_rel)
emas_1, gt_weight_1, ema_low_weight_1, ema_high_weight_1 = train((low_gamma_1, high_gamma_1), 0, update_every=1)
low_gamma_2 = sigma_rel_to_gamma(target_sigma_rel)
high_gamma_2 = sigma_rel_to_gamma(high_sigma_rel)
emas_2, gt_weight_2, ema_low_weight_2, ema_high_weight_2 = train((low_gamma_2, high_gamma_2), 1, update_every=1)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(t, gt_weight_1, label="Original Data", color="gray", alpha=0.7)
plt.plot(t, gt_weight_2, label="Original Data 2", color="gray", alpha=0.7)
plt.plot(t, ema_low_weight_1, label=f"EMA Gamma={low_gamma_1}", color="blue")
plt.plot(t, ema_high_weight_1, label=f"EMA Gamma={high_gamma_1}", color="green")
plt.plot(
    t,
    ema_low_weight_2,
    label=f"EMA Gamma={low_gamma_2} (Ground Truth)",
    color="red",
)

emas_3, gt_weight_3, ema_low_weight_3, ema_high_weight_3 = train((low_gamma_1, high_gamma_1), 2, update_every=10)
plt.plot(t, gt_weight_3, label="Original Data 3", color="gray", alpha=0.7)
plt.plot(t, ema_low_weight_3, label=f"(update 10) EMA Gamma={low_gamma_1}", color="blue", alpha= 0.5, linestyle="--", linewidth=1.5)
plt.plot(t, ema_high_weight_3, label=f"(update 10) EMA Gamma={high_gamma_1}", color="green", alpha=0.5, linestyle="--", linewidth=1.5)

last_approx = emas_1.synthesize_ema_model(gamma = low_gamma_2).ema_model.weight.item()

last_approx_2 = emas_3.synthesize_ema_model(gamma = low_gamma_2).ema_model.weight.item()

plt.scatter(
    t[-1],
    last_approx,
    color="orange",
    marker="x",
    label=f"EMA Gamma={low_gamma_2} Last Approximated",
)

plt.scatter(
    t[-1],
    last_approx_2,
    color="purple",
    marker="x",
    label=f"EMA Gamma={low_gamma_2} Last Approximated (update 10)",
)

plt.title("Power Exponential Moving Average (EMA) Comparison per Gamma and its approximation")
plt.xlabel("Time")
plt.ylabel("Values")
plt.legend()
plt.grid(True)
plt.show()

save_path = "ema_eq.png"
plt.savefig(save_path, dpi=300)



# print(emas_1.synthesize_ema_model(sigma_rel = 0.15).ema_model.weight)
# print(emas_1.synthesize_ema_model(sigma_rel = 0.10).ema_model.weight)
# print(emas_2.synthesize_ema_model(sigma_rel = 0.05).ema_model.weight)
# print(emas_1.ema_models[0].ema_model.weight)
# print(emas_2.ema_models[0].ema_model.weight)

# real_15 = emas_2.ema_models[0].ema_model.weight
# real_05 = emas_1.ema_models[0].ema_model.weight
# real_30 = emas_1.ema_models[1].ema_model.weight
# real_30_2 = emas_2.ema_models[1].ema_model.weight


# fake_15 = emas_1.synthesize_ema_model(sigma_rel = 0.15).ema_model.weight
# fake_10 = emas_1.synthesize_ema_model(sigma_rel = 0.10).ema_model.weight
# fake_10_2 = emas_2.synthesize_ema_model(sigma_rel = 0.10).ema_model.weight
