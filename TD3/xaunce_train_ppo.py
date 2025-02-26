import argparse
import numpy as np
from copy import deepcopy
from xuance.common import get_configs, recursive_dict_update
from xuance.environment import REGISTRY_ENV, make_envs
from xuance.torch.utils.operations import set_seed
from xuance.torch.agents import PPOCLIP_Agent
from my_gazebo_env import MyGazeboEnv  # 导入自定义环境

def parse_args():
    parser = argparse.ArgumentParser("PPO Training for MyGazeboEnv")
    parser.add_argument("--env-id", type=str, default="new-v1")  # 与你的 DDPG 配置一致
    parser.add_argument("--test", type=int, default=0)  # 是否测试模式，0 表示训练
    parser.add_argument("--benchmark", type=int, default=0)  # 是否基准测试，0 表示关闭
    return parser.parse_args()

if __name__ == "__main__":
    # 解析命令行参数
    parser = parse_args()

    # 读取配置文件
    configs_dict = get_configs(file_dir="/home/xzh/Singal_DRL_navigatoion_xuance/TD3/ppo_new_env.yaml")
    configs_dict = recursive_dict_update(configs_dict, parser.__dict__)
    configs = argparse.Namespace(**configs_dict)

    # 设置随机种子
    set_seed(configs.seed)

    # 注册自定义环境
    REGISTRY_ENV[configs.env_name] = MyGazeboEnv

    # 创建环境
    envs = make_envs(configs)

    # 创建 PPO 智能体
    Agent = PPOCLIP_Agent(config=configs, envs=envs)

    # 打印训练信息
    train_information = {"Deep learning toolbox": configs.dl_toolbox,
                         "Calculating device": configs.device,
                         "Algorithm": configs.agent,
                         "Environment": configs.env_name,
                         "Scenario": configs.env_id}
    for k, v in train_information.items():
        print(f"{k}: {v}")

    # 根据模式执行
    if configs.benchmark:
        def env_fn():  # 定义测试环境函数
            configs_test = deepcopy(configs)
            configs_test.parallels = configs_test.test_episode
            return make_envs(configs_test)

        train_steps = configs.running_steps // configs.parallels
        eval_interval = configs.eval_interval // configs.parallels
        test_episode = configs.test_episode
        num_epoch = int(train_steps / eval_interval)

        test_scores = Agent.test(env_fn, test_episode)
        Agent.save_model(model_name="best_model.pth")
        best_scores_info = {"mean": np.mean(test_scores),
                            "std": np.std(test_scores),
                            "step": Agent.current_step}

        for i_epoch in range(num_epoch):
            print("Epoch: %d/%d:" % (i_epoch, num_epoch))
            Agent.train(eval_interval)
            test_scores = Agent.test(env_fn, test_episode)

            if np.mean(test_scores) > best_scores_info["mean"]:
                best_scores_info = {"mean": np.mean(test_scores),
                                    "std": np.std(test_scores),
                                    "step": Agent.current_step}
                Agent.save_model(model_name="best_model.pth")
        print("Best Model Score: %.2f, std=%.2f" % (best_scores_info["mean"], best_scores_info["std"]))
    else:
        if configs.test:
            def env_fn():
                configs.parallels = configs.test_episode
                return make_envs(configs)

            Agent.load_model(path=Agent.model_dir_load)  # 需要指定加载路径
            scores = Agent.test(env_fn, configs.test_episode)
            print(f"Mean Score: {np.mean(scores)}, Std: {np.std(scores)}")
            print("Finish testing.")
        else:
            # 开始训练，与 DDPG 项目逻辑一致
            Agent.train(configs.running_steps // configs.parallels)
            Agent.save_model("final_train_model.pth")
            print("Finish training!")

    # 训练完成
    Agent.finish()