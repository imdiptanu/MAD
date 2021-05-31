# MAD: Multi-task Aggression Detection Framework

### A Multi-Task Learning Framework for Aggression Detection using Transformers


Create a Pyhton 3.7 virtual environment
```
virtualenv --python=python3.7 .
source bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Language Modeler
```cd language-modeler```

Change the language modeling parameters in ```example_config.yaml``` file.

Run the language modeler ```python language_modeler.py --config_yaml example_config.yaml```

## Model Training
```cd model-training```

For now use the Python notebook avilable.

*(TBU - In Progress)*

Change the transformer modeling parameters in ```example_config.yaml``` file.

Training and test transformer modeles ```python model_training.py --config_yaml example_config.yaml```
