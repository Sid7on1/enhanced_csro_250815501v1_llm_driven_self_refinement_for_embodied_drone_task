import os
import logging
from typing import Dict, List
import numpy as np
from abc import ABC, abstractmethod
from torch import load

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfigError(Exception):
    pass

class BaseConfig(ABC):
    def __init__(self):
        self.config = self.load_config()

    @abstractmethod
    def load_config(self) -> Dict:
        pass

    def get(self, key: str, default=None):
        return self.config.get(key, default)

    def set(self, key: str, value):
        self.config[key] = value

    def validate(self):
        required_keys = ["model_path", "environment", "drone_type", "velocity_threshold"]
        for key in required_keys:
            if key not in self.config:
                raise ConfigError(f"Missing required key '{key}' in configuration.")

        if not os.path.isfile(self.config["model_path"]):
            raise ConfigError("Model path is invalid or model file does not exist.")

        if self.config["drone_type"] not in ["industrial", "consumer"]:
            raise ConfigError("Invalid drone type specified in configuration.")

        if not isinstance(self.config["velocity_threshold"], float) or self.config["velocity_threshold"] <= 0:
            raise ConfigError("Velocity threshold must be a positive float value.")

        logger.info("Configuration validated successfully.")

class AgentConfig(BaseConfig):
    def load_config(self) -> Dict:
        return {
            "model_path": "path/to/agent_model.pth",
            "environment": "indoor",
            "drone_type": "industrial",
            "velocity_threshold": 0.5,
            "planning_horizon": 5,
            "planning_frequency": 10,
            "state_evaluation_weight": 0.8,
            "behavior_tree_depth": 3,
            "sensor_update_rate": 30,
            "control_update_rate": 10,
            "simulation_steps": 1000
        }

class EnvironmentConfig(BaseConfig):
    def load_config(self) -> Dict:
        return {
            "map_resolution": 0.1,
            "map_size": (50, 50),
            "obstacle_density": 0.3,
            "goal_distance_threshold": 5.0,
            "goal_angle_threshold": np.pi / 6,
            "initial_position": (10, 10),
            "initial_velocity": (0.0, 0.0),
            "initial_yaw": 0.0,
            "simulation_timestep": 0.1,
            "max_velocity": 2.0,
            "max_yaw_rate": np.pi / 2,
            "control_delay": 0.1,
            "sensor_noise_std": 0.1,
            "process_noise_std": 0.05
        }

def load_agent_model(config: AgentConfig) -> torch.nn.Module:
    model_path = config.get("model_path")
    if not model_path or not os.path.isfile(model_path):
        raise ConfigError("Invalid model path specified in agent configuration.")

    logger.info(f"Loading agent model from: {model_path}")
    model = load(model_path)
    return model

def get_planning_frequency(config: AgentConfig, environment_config: EnvironmentConfig) -> int:
    planning_horizon = config.get("planning_horizon")
    simulation_timestep = environment_config.get("simulation_timestep")
    return int(planning_horizon / simulation_timestep)

def get_control_parameters(config: AgentConfig, environment_config: EnvironmentConfig) -> Dict:
    planning_frequency = get_planning_frequency(config, environment_config)
    control_update_rate = min(planning_frequency, config.get("control_update_rate"))
    return {
        "planning_frequency": planning_frequency,
        "control_update_rate": control_update_rate
    }

def get_simulation_parameters(config: AgentConfig, environment_config: EnvironmentConfig) -> Dict:
    simulation_steps = config.get("simulation_steps")
    control_parameters = get_control_parameters(config, environment_config)
    return {
        "simulation_steps": simulation_steps,
        "planning_frequency": control_parameters["planning_frequency"],
        "control_update_rate": control_parameters["control_update_rate"]
    }

# Example usage
if __name__ == "__main__":
    agent_config = AgentConfig()
    agent_config.validate()

    environment_config = EnvironmentConfig()
    environment_config.validate()

    agent_model = load_agent_model(agent_config)
    control_params = get_control_parameters(agent_config, environment_config)
    simulation_params = get_simulation_parameters(agent_config, environment_config)

    logger.info("Agent and environment configuration loaded successfully.")