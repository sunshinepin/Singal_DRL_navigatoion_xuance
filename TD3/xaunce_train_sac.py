import argparse
import numpy as np
from copy import deepcopy
from xuance.common import get_configs, recursive_dict_update
from xuance.environment import REGISTRY_ENV, make_envs
from xuance.torch.utils.operations import set_seed
from xuance.torch.agents import SAC_Agent
from my_gazebo_env import MyGazeboEnv


def parse_args():
    parser = argparse.ArgumentParser("SAC for MyGazeboEnv")
    parser.add_argument("--env-id", type=str, default="new-v1")
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--benchmark", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    parser = parse_args()
    configs_dict = get_configs(file_dir="/home/xzh/Singal_DRL_navigatoion_xuance/TD3/sac_new_env.yaml")
    configs_dict = recursive_dict_update(configs_dict, parser.__dict__)
    configs = argparse.Namespace(**configs_dict)

    set_seed(configs.seed)
    REGISTRY_ENV[configs.env_name] = MyGazeboEnv
    envs = make_envs(configs)
    Agent = SAC_Agent(config=configs, envs=envs)

    train_information = {
        "Deep learning toolbox": configs.dl_toolbox,
        "Calculating device": configs.device,
        "Algorithm": configs.agent,
        "Environment": configs.env_name,
        "Scenario": configs.env_id
    }
    
    for k, v in train_information.items():
        print(f"{k}: {v}")


    if configs.benchmark:
        def env_fn():
            configs_test = deepcopy(configs)
            configs_test.parallels = configs_test.test_episode
            return make_envs(configs_test)

        train_steps = configs.running_steps // configs.parallels
        eval_interval = configs.eval_interval // configs.parallels
        test_episode = configs.test_episode
        num_epoch = int(train_steps / eval_interval)

        test_scores = Agent.test(env_fn, test_episode)
        Agent.save_model("best_model.pth")  # 仅使用 model_name 参数
        best_scores_info = {
            "mean": np.mean(test_scores),
            "std": np.std(test_scores),
            "step": Agent.current_step
        }

        for i_epoch in range(num_epoch):
            print(f"Epoch: {i_epoch}/{num_epoch}:")
            Agent.train(eval_interval)
            test_scores = Agent.test(env_fn, test_episode)

            if np.mean(test_scores) > best_scores_info["mean"]:
                best_scores_info = {
                    "mean": np.mean(test_scores),
                    "std": np.std(test_scores),
                    "step": Agent.current_step
                }
                Agent.save_model("best_model.pth")  # 仅使用 model_name 参数

        print(f"Best Model Score: {best_scores_info['mean']:.2f}, std={best_scores_info['std']:.2f}")
    else:
        if configs.test:
            def env_fn():
                configs.parallels = configs.test_episode
                return make_envs(configs)

            Agent.load_model(configs.model_dir)  # load_model 使用 model_dir
            scores = Agent.test(env_fn, configs.test_episode)
            print(f"Mean Score: {np.mean(scores)}, Std: {np.std(scores)}")
            print("Finish testing.")
        else:
            Agent.train(configs.running_steps // configs.parallels)
            Agent.save_model("final_train_model.pth")  # 仅使用 model_name 参数
            print("Finish training!")

    Agent.finish()