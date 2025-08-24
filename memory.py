import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict
from threading import Lock

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MEMORY_SIZE = 10000
BATCH_SIZE = 32
EPOCHS = 10

# Enum for memory types
class MemoryType(Enum):
    EXPERIENCE = 1
    TRANSITION = 2

# Dataclass for experience
@dataclass
class Experience:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool

# Dataclass for transition
@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool

# Memory class
class Memory:
    def __init__(self, memory_size: int = MEMORY_SIZE):
        self.memory_size = memory_size
        self.memory = []
        self.lock = Lock()

    def add_experience(self, experience: Experience):
        with self.lock:
            self.memory.append(experience)
            if len(self.memory) > self.memory_size:
                self.memory.pop(0)

    def add_transition(self, transition: Transition):
        with self.lock:
            self.memory.append(transition)
            if len(self.memory) > self.memory_size:
                self.memory.pop(0)

    def get_experiences(self, batch_size: int = BATCH_SIZE) -> List[Experience]:
        with self.lock:
            experiences = self.memory[:batch_size]
            self.memory = self.memory[batch_size:]
            return experiences

    def get_transitions(self, batch_size: int = BATCH_SIZE) -> List[Transition]:
        with self.lock:
            transitions = self.memory[:batch_size]
            self.memory = self.memory[batch_size:]
            return transitions

# Dataset class for experience replay
class ExperienceReplayDataset(Dataset):
    def __init__(self, memory: Memory, batch_size: int = BATCH_SIZE):
        self.memory = memory
        self.batch_size = batch_size

    def __len__(self):
        return len(self.memory.memory)

    def __getitem__(self, idx: int):
        experience = self.memory.memory[idx]
        return {
            "state": experience.state,
            "action": experience.action,
            "reward": experience.reward,
            "next_state": experience.next_state,
            "done": experience.done
        }

# Data loader for experience replay
class ExperienceReplayDataLoader:
    def __init__(self, dataset: ExperienceReplayDataset):
        self.dataset = dataset

    def get_data_loader(self):
        return DataLoader(self.dataset, batch_size=self.dataset.batch_size, shuffle=True)

# Agent memory class
class AgentMemory:
    def __init__(self):
        self.memory = Memory()
        self.data_loader = None

    def add_experience(self, experience: Experience):
        self.memory.add_experience(experience)

    def add_transition(self, transition: Transition):
        self.memory.add_transition(transition)

    def get_experiences(self, batch_size: int = BATCH_SIZE) -> List[Experience]:
        return self.memory.get_experiences(batch_size)

    def get_transitions(self, batch_size: int = BATCH_SIZE) -> List[Transition]:
        return self.memory.get_transitions(batch_size)

    def get_data_loader(self):
        if not self.data_loader:
            dataset = ExperienceReplayDataset(self.memory, batch_size=BATCH_SIZE)
            self.data_loader = ExperienceReplayDataLoader(dataset)
        return self.data_loader.get_data_loader()

# Unit tests
import unittest

class TestMemory(unittest.TestCase):
    def test_add_experience(self):
        memory = Memory()
        experience = Experience(np.array([1, 2, 3]), 1, 1.0, np.array([4, 5, 6]), False)
        memory.add_experience(experience)
        self.assertEqual(len(memory.memory), 1)

    def test_add_transition(self):
        memory = Memory()
        transition = Transition(np.array([1, 2, 3]), 1, 1.0, np.array([4, 5, 6]), False)
        memory.add_transition(transition)
        self.assertEqual(len(memory.memory), 1)

    def test_get_experiences(self):
        memory = Memory()
        experience1 = Experience(np.array([1, 2, 3]), 1, 1.0, np.array([4, 5, 6]), False)
        experience2 = Experience(np.array([7, 8, 9]), 2, 2.0, np.array([10, 11, 12]), False)
        memory.add_experience(experience1)
        memory.add_experience(experience2)
        experiences = memory.get_experiences()
        self.assertEqual(len(experiences), 2)

    def test_get_transitions(self):
        memory = Memory()
        transition1 = Transition(np.array([1, 2, 3]), 1, 1.0, np.array([4, 5, 6]), False)
        transition2 = Transition(np.array([7, 8, 9]), 2, 2.0, np.array([10, 11, 12]), False)
        memory.add_transition(transition1)
        memory.add_transition(transition2)
        transitions = memory.get_transitions()
        self.assertEqual(len(transitions), 2)

if __name__ == "__main__":
    unittest.main()