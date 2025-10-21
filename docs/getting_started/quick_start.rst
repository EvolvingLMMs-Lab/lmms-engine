Quick Start
===========

Get up and running with LMMs Engine in minutes.

Installation
------------

.. code-block:: bash

   git clone <repository-url>
   cd lmms-engine-mini
   pip install -e .

Your First Training Job
-----------------------

1. **Prepare your data**: See :doc:`../user_guide/data_prep` for detailed instructions
2. **Configure your dataset**: See :doc:`../user_guide/datasets` for configuration options
3. **Run training**: See :doc:`train` for detailed training instructions

Basic Configuration
--------------------

LMMs Engine uses YAML configuration files. Here's a minimal example:

.. code-block:: yaml

   model_config:
     model_type: "your_model"
     pretrained_model_name_or_path: "path/to/model"
   
   dataset_config:
     dataset_type: "your_dataset"
     data_path: "/path/to/data"
   
   trainer_args:
     output_dir: "./output"
     num_train_epochs: 3
     per_device_train_batch_size: 8

Next Steps
----------

- Explore :doc:`../user_guide/index` for detailed guides
- Check :doc:`../reference/video_configuration` if working with video data
- Review :doc:`../reference/design_principle` to understand the architecture
