

# Design Principle

The main design principle of the framework is using following concept
1. Factory Pattern
2. Builder

## Factory Pattern
Factory pattern is simple. Basically is just using if else to create component we want. But what it nice is that it encapsulate most of the building process we want under the hood and is good if we want to use it as a separate api. So it might be ugly inside the factory (which make sense), but is easy for us to create new factory and build component all around the framework as we want.

Currently the factory mainly responsible for building these components
1. Model
2. Trainer (But anyway we only have 1 trainer that actually can run now)
3. Datasets
4. Processor

With different factories, we are able to combine different component as easy as we want with out adding in ugly if else inside our main logic. 

Yeah encapsulation is good. At least we don't have to look into the ugly logic every time we debug.


## Builder
So the idea of the builder is basically our object actually builds the component based on our needs and only we we actually needs the object. This allow us to have a lightweighted training framework with flexible plugin to be turn on and off. For example, in the trainer object, we do not pass in all the model and dataset at all. Instead, we initialize our trainer with a separate config object. Then, when we actually starts training. We do the building
```python
def build(self):
    self.model = self._build_model()
    self.train_dataset = self._build_train_dataset()
    if self.model_config.pretrain_mm_mlp_adapter is not None:
        self._load_mm_projector()
    if self.config.trainer_args.use_liger_kernel:
        self._apply_linger_kernel()
        # Set to False as we already apply the liger kernel by ourselves
        self.config.trainer_args.use_liger_kernel = False
``` 

This allow us to separate the building process of different components and each building process and be determine by the config.