import gymnasium as gym, torch, time
from stable_baselines3 import PPO

# --- AI network evaluated every physics step --------------------------
net = torch.nn.Sequential(
    torch.nn.Linear(4, 32),
    torch.nn.Tanh(),
    torch.nn.Linear(32, 2)
)
print("PyTorch inference each frame")
obs_tensor = torch.zeros(4)
# ----------------------------------------------------------------------

env = gym.make("CartPole-v1", render_mode="human")

# ----- quick PPO training (~30 s CPU) --------------------------
model = PPO("MlpPolicy", env,
            n_steps=128, batch_size=128, verbose=0)
print("PPO training 5 000 timesteps …")
model.learn(total_timesteps=5000)
print("PPO training done")
# ---------------------------------------------------------------

obs, _ = env.reset()
start = time.time()


while time.time() - start < 60:          # run exactly 60 s
    obs_tensor[:] = torch.tensor(obs)    # copy state to tensor
    action, _ = model.predict(obs, deterministic=True)
    obs, _, done, trunc, _ = env.step(action)        # physics step
    if done or trunc:                                # pole fell → reset
        obs, _ = env.reset()

env.close()                                           # closes window


