#!/usr/bin/env python3
"""
task_registry.py  ── Registry and factory for Task instances in agent pipelines
Author  : [Zhou Liu]
License : MIT
Created : 2024-07-02

This module provides a TaskRegistry class for managing and instantiating Task objects by name.

Features:
* Decorator-based registration of task factories.
* Retrieval of Task instances by name, with support for custom prompt templates.
* Listing of all registered tasks.
* Ensures unique task names and clear error handling for unregistered tasks.

Intended for extensible agent pipelines where tasks are defined, registered, and instantiated dynamically.

Thread-safety: Registry modification is not thread-safe by default and should be synchronized if used in concurrent environments.
"""
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