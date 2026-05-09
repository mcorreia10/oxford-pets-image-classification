# Pet Breed Classifier - Streamlit App

This app loads a trained PyTorch model and predicts the pet breed from an uploaded image.

## Required files

Place trained model weights inside the `models/` folder:

```text
models/best_convnext.pth
models/best_vit.pth
models/best_cnn.pth
```

Also place `class_names.json` in the project root. Export it from the training notebook with:

```python
import json
with open("class_names.json", "w") as f:
    json.dump(train_dataset.classes, f)
```

The class order must match the training dataset exactly.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Important notes

- The image preprocessing uses ImageNet normalization.
- The CustomCNN class in `app.py` must match the exact architecture used during training.
- For the final demo, ConvNeXt is recommended because it achieved the strongest test performance.
