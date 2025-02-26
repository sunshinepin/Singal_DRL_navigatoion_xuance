import argparse
from xuance.common import get_configs
from xuance.environment import REGISTRY_ENV
from xuance.environment import make_envs
from xuance.torch.agents import DDPG_Agent
from my_gazebo_env import MyGazeboEnv  # 导入自定义环境

# 读取配置文件
configs_dict = get_configs(file_dir="/home/xzh/drl_navigation_signal/DRL_navigatoion_xuance/DRL-robot-navigation-main/TD3/ddpg_new_env.yaml")
configs = argparse.Namespace(**configs_dict)

# 注册自定义环境
REGISTRY_ENV[configs.env_name] = MyGazeboEnv

# 创建环境
envs = make_envs(configs)  
Agent = DDPG_Agent(config=configs, envs=envs)  # 创建 DDPG 智能体

# 开始训练
Agent.train(configs.running_steps // configs.parallels)

# 保存模型
Agent.save_model("final_train_model.pth")

# 训练完成
Agent.finish()
