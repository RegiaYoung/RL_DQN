from collections import deque
import os
import random
from tqdm import tqdm

import argparse
import torch

from utils_drl import Agent
from utils_env import MyEnv
from utils_memory import ReplayMemory

def main():
    GAMMA = args["gamma"]
    GLOBAL_SEED = args["seed"]
    MEM_SIZE = args["memory_size"]
    RENDER = args["render"]
    SAVE_PREFIX = args["save_folder"]
    LOAD_FILE = args["load_folder"]
    STACK_SIZE = args["stack_size"]

    EPS_START = args["eps_start"]
    EPS_END = args["eps_end"]
    EPS_DECAY = args["eps_decay"]

    BATCH_SIZE = args["batch_size"]
    POLICY_UPDATE = args["policy_update"]
    TARGET_UPDATE = args["target_update"]
    WARM_STEPS = args["warm_steps"]
    MAX_STEPS = args["max_steps"]
    EVALUATE_FREQ = args["evaluate_freq"]
    CUDA_DEVICE = args["cuda_device"]
    REW_PATH = args["rew_path"]

    torch.cuda.set_device(CUDA_DEVICE)

    rand = random.Random()
    rand.seed(GLOBAL_SEED)
    new_seed = lambda: rand.randint(0, 1_000_000)
    os.mkdir(SAVE_PREFIX)

    torch.manual_seed(new_seed())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = MyEnv(device)
    agent = Agent(
        env.get_action_dim(),
        device,
        GAMMA,
        new_seed(),
        EPS_START,
        EPS_END,
        EPS_DECAY,
        LOAD_FILE
    )
    memory = ReplayMemory(STACK_SIZE + 1, MEM_SIZE, device)

    #### Training ####
    obs_queue: deque = deque(maxlen=5)
    done = True

    progressive = tqdm(range(MAX_STEPS), total=MAX_STEPS,
                    ncols=50, leave=False, unit="b")
    for step in progressive:
        if done:
            observations, _, _ = env.reset()
            for obs in observations:
                obs_queue.append(obs)

        training = len(memory) > WARM_STEPS
        state = env.make_state(obs_queue).to(device).float()
        action = agent.run(state, training)
        obs, reward, done = env.step(action)
        obs_queue.append(obs)
        memory.push(env.make_folded_state(obs_queue), action, reward, done)

        if step % POLICY_UPDATE == 0 and training:
            agent.learn(memory, BATCH_SIZE)

        if step % TARGET_UPDATE == 0:
            agent.sync()

        if step % EVALUATE_FREQ == 0:
            avg_reward, frames = env.evaluate(obs_queue, agent, render=RENDER)
            with open(REW_PATH, "a") as fp:
                fp.write(f"{step//EVALUATE_FREQ:3d} {step:8d} {avg_reward:.1f}\n")
            if RENDER:
                prefix = f"eval_{step//EVALUATE_FREQ:03d}"
                os.mkdir(prefix)
                for ind, frame in enumerate(frames):
                    with open(os.path.join(prefix, f"{ind:06d}.png"), "wb") as fp:
                        frame.save(fp, format="png")
            agent.save(os.path.join(
                SAVE_PREFIX, f"model_{step//EVALUATE_FREQ:03d}"))
            done = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma",type=float,nargs="?",default=0.99,help="decay rate")
    parser.add_argument("--seed",type=int,nargs="?",default=0,help="the seed to init weight")
    parser.add_argument("--memory_size",type=int,nargs="?",default=100_000,help="store the memory")
    parser.add_argument("--stack_size",type=int,nargs="?",default=4,help="store the relative frame's num")
    parser.add_argument("--eps_start",type=float,nargs="?",default=1.,help="start of the eps")
    parser.add_argument("--eps_end",type=float,nargs="?",default=0.1,help="end of the eps")
    parser.add_argument("--eps_decay",type=float,nargs="?",default=1_000_000,help="decay rate to run")
    parser.add_argument("--render",type=bool,nargs="?",default=False,help="render the training")

    parser.add_argument("--batch_size",type=int,nargs="?",default=32,help="every train batch's size")
    parser.add_argument("--policy_update",type=int,nargs="?",default=4,help="the freq to update the policy DQN")
    parser.add_argument("--target_update",type=int,nargs="?",default=10_000,help="the freq to update the target DQN")
    parser.add_argument("--warm_steps",type=int,nargs="?",default=50_000,help="warm the agent and not to train")
    parser.add_argument("--max_steps",type=int,nargs="?",default=50_000_000,help="the all steps")
    parser.add_argument("--evaluate_freq",type=int,nargs="?",default=100_000,help="the freq to evaluate and store the model")


    parser.add_argument("--save_folder",type=str,nargs="?",default="./models",help="where to store the evaluate model")
    parser.add_argument("--load_folder",type=str,nargs="?",default=None,help="where to load the pre-trained model")
    parser.add_argument("--cuda_device",type=int,nargs="?",default=0,help="use which GPU")
    parser.add_argument("--rew_path",type=str,nargs="?",default="rewards.txt",help="the path of rewards.txt")
    args = vars(parser.parse_args())
    main()
