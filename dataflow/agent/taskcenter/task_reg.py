from typing import Callable, Dict
from ..promptstemplates.prompt_template import PromptsTemplateGenerator
from .task_dispatcher import Task

class TaskRegistry:
    """
    Factory for all Task instances, supporting retrieval by name.
    """
    _factories: Dict[str, Callable[[PromptsTemplateGenerator], Task]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(factory: Callable[[PromptsTemplateGenerator], Task]):
            if name in cls._factories:
                raise KeyError(f"Task '{name}' is already registered")
            cls._factories[name] = factory
            return factory
        return decorator

    @classmethod
    def get(cls, name: str, prompts_template: PromptsTemplateGenerator) -> Task:
        factory = cls._factories.get(name)
        if not factory:
            raise KeyError(f"No task named '{name}' was found. Please register it first.")
        return factory(prompts_template)

    @classmethod
    def list_tasks(cls):
        return list(cls._factories.keys())