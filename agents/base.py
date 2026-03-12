from abc import ABC, abstractmethod


class BaseAgent(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def select_action(self, state: int, info: dict = None) -> int:
        pass

    @abstractmethod
    def learn(self, state: int, action: int, reward: float,
              next_state: int, done: bool, info: dict = None) -> None:
        pass

    @abstractmethod
    def get_params(self) -> dict:
        pass
