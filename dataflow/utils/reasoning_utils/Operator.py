from abc import ABC, abstractmethod


class Operator(ABC):

    @abstractmethod
    def check_config(self, config: dict) -> None:
        """
        Check the config of the operator. If config lacks any required keys, raise an error.
        """
        pass
    
    @abstractmethod
    def run(self) -> None:
        """
        Main function to run the operator.
        """
        pass