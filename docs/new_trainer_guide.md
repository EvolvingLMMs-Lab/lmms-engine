### Trainer Registry: Adding a new trainer

This guide explains how to define, register, and use a custom trainer via the centralized registry used by the `TrainRunner`.

### How the registry works

- Registration is decorator-driven: you attach a key to your class/function using `@TRAINER_REGISTER.register(...)`.
- Lookup happens in `TrainRunner._build_trainer` via `TRAINER_REGISTER[self.config.trainer_type]`.
- The trainer class is then instantiated with specific keyword arguments (see below).

### Constructor requirements

`TrainRunner` will instantiate your trainer like this:

```python
trainer_cls(
    model=self.model,
    args=self.config.trainer_args,
    data_collator=self.train_dataset.get_collator(),
    train_dataset=self.train_dataset,
    eval_dataset=self.eval_dataset,
    processing_class=self.train_dataset.processor,
)
```

Your trainer should accept these keyword arguments (unused ones can be ignored via `**kwargs`). If you subclass Hugging Face `transformers.Trainer`, note that it expects `tokenizer=`; in our stack, we pass `processing_class=`. You can forward `processing_class` to the appropriate place (e.g., `tokenizer`) inside your constructor.

### Step 1: Implement your trainer

```python
# src/lmms_engine/train/my_trainer.py
from transformers import Trainer as HFTrainer
from lmms_engine.train.registry import TRAINER_REGISTER

@TRAINER_REGISTER.register("my_trainer")  # or omit the string to use the class name as the key
class MyTrainer(HFTrainer):
    def __init__(
        self,
        *,
        model,
        args,
        data_collator,
        train_dataset,
        eval_dataset=None,
        processing_class=None,
        **kwargs,
    ):
        # If subclassing HF Trainer, you can map processing_class to tokenizer
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=processing_class,
        )
        # ... any custom init logic ...
```

Notes:
- Using `@TRAINER_REGISTER.register("my_trainer")` registers the key as `"my_trainer"`.
- If you use `@TRAINER_REGISTER.register` without parentheses, the key defaults to the class name (`MyTrainer`).
- Re-registering an existing key will overwrite the previous value and print a warning.

### Step 2: Ensure registration is imported

Registration happens at import time. Make sure the module containing your decorator is imported before building the trainer.

Common options:
- Import in `src/lmms_engine/train/__init__.py`:

```python
# src/lmms_engine/train/__init__.py
from . import my_trainer  # noqa: F401 ensures registration side-effect
```

- Or import explicitly in your application/runner setup prior to calling `TrainRunner.build()`.

### Step 3: Select your trainer in config

Set `trainer_type` in your config to the registry key you used in registration.

```yaml
# examples/load_from_config_example.yaml (snippet)
trainer_type: my_trainer
trainer_args:
  output_dir: ./output/run
  bf16: true
  # ... other args ...
```

If you registered without a string, use the class name instead (e.g., `trainer_type: MyTrainer`).

### Step 4: Run

`TrainRunner` will resolve the trainer class from the registry and instantiate it with the expected arguments.

### Troubleshooting

- KeyError: Ensure your module with the decorator ran before `TrainRunner` builds the trainer (import ordering).
- Constructor errors: Ensure your `__init__` accepts the arguments listed above; capture extras with `**kwargs` if needed.
- Duplicate key warning: Another module registered the same key; either change the key or keep only one registration.


