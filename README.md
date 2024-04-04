# Chess-Evaluator
Computer Vision that detects chess boards and evaluates position using Stock Fish. Uses Faster R-CNN model with a ResNet-50-FPN backbone.



# Learning model
Get dataset to folder `model_chess_training/datasets/` and set correct `PATH_TO_DATASETS` in `model_chess_training/config.py`

```sh
python3 model_chess_training/train_chess_detection.py
```

# Run
```sh
python3 main.py
```

## Example

![alt text](docs\example.png "Example")