Introduction
=============

LMMs Engine is a flexible and extensible framework designed for training large multimodal models. It provides:

- **Modular Architecture**: Built on factory patterns, making it easy to extend with custom components
- **Flexible Training**: Support for various model architectures, datasets, and training strategies
- **Configuration-Driven**: Lightweight framework with configuration-based component building
- **Video Support**: Native support for video processing with multiple backend options
- **Scalable**: Designed to handle large-scale training with performance optimization

Getting Started
---------------

To begin using LMMs Engine, proceed to the :doc:`quick_start` guide or explore the :doc:`../user_guide/index` for detailed information about specific features.

Key Concepts
------------

- **Factory Pattern**: Used to encapsulate component creation for models, trainers, datasets, and processors
- **Builder Pattern**: Lazy initialization of components, loading them only when needed
- **MVC Architecture**: Clean separation of concerns for future extensibility

For more details about the architecture, see the :doc:`../reference/design_principle`.
