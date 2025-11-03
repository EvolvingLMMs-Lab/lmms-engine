Adding a Custom Parallel Strategy (Expert Parallel)
=====================================================

This guide walks you through implementing a custom parallel strategy in LMMs Engine using **Qwen3-MoE** as a concrete example. We'll cover expert parallelism, which is particularly useful for mixture-of-experts (MoE) models.

.. important::
   Custom parallelization strategies can only be used with the **FSDP2 trainer**. Ensure your training configuration uses FSDP2 for distributed training.

Overview
--------

Expert parallelism distributes MoE experts across multiple devices to reduce memory pressure and improve throughput. The implementation involves five key steps:

1. Create the model folder in the parallel directory
2. Define a parallel style class
3. Implement parallelization functions
4. Apply FSDP (Fully Sharded Data Parallel)
5. Register the parallelization function

Step 1: Create the Model Folder
-------------------------------

Create a new directory for your custom parallel strategy under ``src/lmms_engine/parallel/``:

.. code-block:: bash

   mkdir -p src/lmms_engine/parallel/qwen3_moe

Create two main files:
- ``style.py`` - Define your parallel style
- ``parallelize.py`` - Implement parallelization functions
- ``__init__.py`` - Export public interfaces

Step 2: Define Your Parallel Style
-----------------------------------

The parallel style class extends PyTorch's ``ParallelStyle`` and defines how to distribute parameters across the device mesh.

**File: ``src/lmms_engine/parallel/qwen3_moe/style.py``**

Key components:

.. code-block:: python

   from torch.distributed.tensor.parallel import ParallelStyle
   from torch.distributed.tensor import DeviceMesh, Shard, Replicate, DTensor
   
   class Qwen3MoeParallelStyle(ParallelStyle):
       """
       Custom parallel style for Qwen3MoE expert parallelism.
       Handles token dispatching and combining across expert parallel ranks.
       """
       
       def __init__(
           self,
           input_layouts: Optional[Placement] = None,
           output_layouts: Optional[Placement] = None,
           use_local_output: bool = True,
       ) -> None:
           super().__init__()
           # Define how inputs/outputs should be distributed
           self.input_layouts = (input_layouts or Shard(0),)
           self.output_layouts = (output_layouts or Shard(0),)
           self.use_local_output = use_local_output

The parallel style must implement these key methods:

- **``_input_fn``**: Preprocesses inputs before they reach the module (token dispatching)
- **``_output_fn``**: Postprocesses outputs after the module (token combining)
- **``_partition_fn``**: Distributes module parameters across the device mesh
- **``_apply``**: Applies the parallel style to the module

Example from Qwen3MoE:

.. code-block:: python

   @staticmethod
   def _partition_fn(name, mod, device_mesh):
       """Distribute expert parameters across the device mesh."""
       if isinstance(mod, Qwen3MoeExperts):
           expert_parallel_dim = 0
           
           # Distribute each expert parameter
           mod.register_parameter(
               "up_proj",
               nn.Parameter(
                   distribute_tensor(
                       mod.up_proj,
                       device_mesh,
                       [Shard(expert_parallel_dim)],
                   )
               ),
           )

The ``_input_fn`` and ``_output_fn`` handle token routing:

.. code-block:: python

   def _input_fn(self, inputs, mesh: DeviceMesh):
       """
       Dispatch tokens to appropriate experts.
       Uses all-to-all communication for token exchange.
       """
       routed_input, num_tokens_per_expert = inputs
       
       if ep_world_size > 1:
           # Perform token dispatching across expert parallel group
           (
               routed_input,
               input_splits,
               output_splits,
               num_tokens_per_expert_group,
           ) = _token_dispatch(routed_input, num_tokens_per_expert)
           # Store metadata for output combining
           self.input_splits = input_splits
           self.output_splits = output_splits
       
       return routed_input
   
   def _output_fn(self, output, mesh: DeviceMesh):
       """Combine outputs from all experts back to original token order."""
       if ep_world_size > 1:
           output = _token_combine(output, self.input_splits, self.output_splits)
       return output

Step 3: Implement Parallelization Functions
---------------------------------------------

**File: ``src/lmms_engine/parallel/qwen3_moe/parallelize.py``**

This file contains the main parallelization logic:

.. code-block:: python

   from torch.distributed.tensor.parallel import parallelize_module
   
   def apply_qwen3_moe_parallel(
       model: Qwen3MoeForCausalLM,
       ep_mesh: DeviceMesh,
       tp_mesh: DeviceMesh = None,
       **kwargs,
   ):
       """
       Apply expert parallel style to model layers.
       
       Args:
           model: The model to parallelize
           ep_mesh: Expert parallel device mesh
           tp_mesh: Tensor parallel device mesh (if supported)
       """
       for decoder_layer in model.model.layers:
           module = decoder_layer.mlp
           ep_plan = Qwen3MoeParallelStyle()
           
           parallelize_module(
               module.experts,
               device_mesh=ep_mesh,
               parallelize_plan=ep_plan,
           )

The key function is ``apply_qwen3_moe_parallelize_fn`` which orchestrates the entire process:

.. code-block:: python

   def apply_qwen3_moe_parallelize_fn(
       model: Qwen3MoeForCausalLM,
       train_args: TrainingArguments,
       **kwargs,
   ):
       """
       Main entry point for applying parallelization to Qwen3MoE.
       This function is called by the training system.
       """
       ep_size = process_group_manager.ep_size
       
       # Step 1: Stack expert parameters into efficient format
       stack_expert_params(model)
       full_state_dict = model.state_dict()
       
       # Step 2: Apply expert parallel style (if ep_size > 1)
       if ep_size > 1:
           ep_mesh = process_group_manager.device_mesh["ep"]
           apply_qwen3_moe_parallel(model, ep_mesh=ep_mesh, **kwargs)
       
       # Step 3: Apply FSDP2
       apply_qwen3_moe_fsdp2(model, train_args, **kwargs)
       
       # Step 4: Restore full state dict
       fsdp2_load_full_state_dict(model, full_state_dict)

Step 4: Apply FSDP2
--------------------

After defining the parallel style, apply **Fully Sharded Data Parallel (FSDP2)** for distributed training:

.. code-block:: python

   from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
   
   def apply_qwen3_moe_fsdp2(
       model: Qwen3MoeForCausalLM,
       train_args: TrainingArguments,
       **kwargs,
   ):
       """Apply FSDP2 sharding to model."""
       
       # Configure mixed precision
       mp_policy = MixedPrecisionPolicy(
           param_dtype=torch.bfloat16,
           reduce_dtype=torch.bfloat16,
           output_dtype=torch.float32,
       )
       
       dp_mesh = process_group_manager.device_mesh["fsdp"]
       
       fsdp_kwargs = {
           "reshard_after_forward": True,
           "mp_policy": mp_policy,
           "mesh": dp_mesh,
       }
       
       # Shard expert layers
       ep_size = process_group_manager.ep_size
       if ep_size > 1:
           # Use a different mesh for experts
           expert_fsdp_kwargs = dict(fsdp_kwargs)
           expert_fsdp_kwargs["mesh"] = process_group_manager.device_mesh["dp_shard_mod_ep"]
           expert_fsdp_kwargs["shard_placement_fn"] = lambda p: Shard(1)
           fully_shard(decoder_layer.mlp, **expert_fsdp_kwargs)
       
       # Shard other layers
       for decoder_layer in model.model.layers:
           fully_shard(decoder_layer.self_attn, **fsdp_kwargs)
       
       fully_shard(model.model.embed_tokens, **fsdp_kwargs)
       fully_shard(model, **fsdp_kwargs)

Step 5: Register Your Parallelization Function
-----------------------------------------------

Register your parallelization function in the main registry so it can be discovered by the training system.

**File: ``src/lmms_engine/parallel/parallelize.py``**

.. code-block:: python

   from .qwen3_moe.parallelize import apply_qwen3_moe_parallelize_fn
   
   MODEL_TO_PARALLEL_METHOD = {
       "qwen3_moe": apply_qwen3_moe_parallelize_fn,
   }
   
   def apply_parallelize(model, model_type, train_args: TrainingArguments, **kwargs):
       """
       Apply parallelization based on model type.
       
       Args:
           model: The model to parallelize
           model_type: Key in MODEL_TO_PARALLEL_METHOD (e.g., "qwen3_moe")
           train_args: Training configuration
       
       Raises:
           ValueError: If model_type is not supported
       """
       if model_type not in MODEL_TO_PARALLEL_METHOD:
           raise ValueError(f"Model type {model_type} not supported")
       
       return MODEL_TO_PARALLEL_METHOD[model_type](model, train_args, **kwargs)

**File: ``src/lmms_engine/parallel/qwen3_moe/__init__.py``**

.. code-block:: python

   from .parallelize import apply_qwen3_moe_parallel
   from .style import Qwen3MoeParallelStyle
   
   __all__ = ["apply_qwen3_moe_parallel", "Qwen3MoeParallelStyle"]

Key Utilities
-------------

Helper functions for token routing are typically placed in a shared utils module:

**File: ``src/lmms_engine/parallel/expert_parallel/utils.py``**

- ``_token_dispatch``: Distributes tokens to experts across ranks using all-to-all communication
- ``_token_combine``: Combines tokens back from all experts  
- ``_compute_permute_indices``: Computes token permutation indices for efficient routing

These functions handle the communication and coordination between expert parallel ranks.

Configuration in Training Arguments
------------------------------------

To enable expert parallelism in your training config:

.. code-block:: yaml

   ep_degree: 8

Summary
-------

To add a custom parallel strategy:

1. ✅ Create a folder under ``src/lmms_engine/parallel/`` with your model name
2. ✅ Define a ``ParallelStyle`` subclass that implements ``_partition_fn``, ``_input_fn``, ``_output_fn``, and ``_apply``
3. ✅ Implement parallelization functions in ``parallelize.py``
4. ✅ Apply FSDP2 sharding using ``fully_shard`` with appropriate device meshes
5. ✅ Register your parallelization function in ``src/lmms_engine/parallel/parallelize.py``

Remember: Custom parallelization **only works with the FSDP2 trainer**. The parallelization is applied automatically when the training system detects your model type.

For questions about token routing, device meshes, or FSDP2 configuration, refer to the PyTorch Distributed documentation and the LMMs Engine API reference.
