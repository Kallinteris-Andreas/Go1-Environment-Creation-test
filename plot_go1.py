import numpy as np
import matplotlib.pyplot as plt
import stable_baselines3

RUNS = 1  # Number of statistical runs
FILE_NAME = "figures/go1"

steps = np.load('results/go1/TD3/run_0/evaluations.npz')['timesteps']

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

for algorithm in ["SAC"]:
    returns = np.average(np.array([np.load(f'results/go1/{algorithm}/run_{run}/evaluations.npz')['results'][:steps.size] for run in range(RUNS)]), axis=2)
    returns_len = np.average(np.array([np.load(f'results/go1/{algorithm}/run_{run}/evaluations.npz')['ep_lengths'][:steps.size] for run in range(RUNS)]), axis=2)

    ax.plot(steps, np.average(returns, axis=0), label=f'go1 {algorithm}')
    ax.fill_between(steps, np.min(returns, axis=0), np.max(returns, axis=0), alpha=0.2)


ax.set_title(f'SB3 v{stable_baselines3.__version__} on Gymnasium/MuJoCo/ with Go1, for ' + str(RUNS) + ' Runs, episodic return')
ax.legend()

fig.set_figwidth(16)
fig.set_figheight(9)

plt.savefig(FILE_NAME + ".eps", bbox_inches="tight")
plt.savefig(FILE_NAME + ".png", bbox_inches="tight")
