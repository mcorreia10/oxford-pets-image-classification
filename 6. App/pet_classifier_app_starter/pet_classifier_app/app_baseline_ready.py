import json
from pathlib import Path

import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from transformers import AutoModelForImageClassification


APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
CLASS_NAMES_PATH = APP_DIR / "class_names.json"

st.set_page_config(page_title="Pet Breed Classifier", page_icon="🐾", layout="centered")

# -----------------------------
# Configuration
# -----------------------------
MODEL_CONFIG = {
    "Baseline CNN": {
        "path": MODELS_DIR / "oxford_pet_simplecnn_best.pth",
        "type": "cnn",
    },
        "ResNet": {
        "path": MODELS_DIR / "ResNet18_03_lr1e4_bs16_layer4_fc_best.pth",
        "type": "resnet",
    },
    "ViT": {
        "path": MODELS_DIR / "exp3_vit_last_block_best.pth",
        "type": "vit",
    },
    "ConvNeXt": {
        "path": MODELS_DIR / "exp2_convnext_img256_best.pth",
        "type": "convnext",
    },
}

DOG_CLASSES = {
    "american_bulldog", "american_pit_bull_terrier", "basset_hound", "beagle",
    "boxer", "chihuahua", "english_cocker_spaniel", "english_setter",
    "german_shorthaired", "great_pyrenees", "havanese", "japanese_chin",
    "keeshond", "leonberger", "miniature_pinscher", "newfoundland", "pomeranian",
    "pug", "saint_bernard", "samoyed", "scottish_terrier", "shiba_inu",
    "staffordshire_bull_terrier", "wheaten_terrier", "yorkshire_terrier",
}


# -----------------------------
# Model definitions
# -----------------------------
class BaselineCNN(nn.Module):
    """
    Final baseline CNN architecture from the training notebook.

    This must match the architecture used to save:
    models/oxford_pet_simplecnn_best.pth

    Training configuration:
    - IMG_SIZE = 160
    - dropout = 0.4
    - num_classes = 37
    - 4 convolutional blocks
    - 2 Conv2d layers per block
    - BatchNorm + ReLU + MaxPool
    - AdaptiveAvgPool2d + Linear head
    """

    def __init__(self, num_classes: int = 37, dropout: float = 0.4):
        super().__init__()

        def block(in_c: int, out_c: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),

                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),

                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, 32),
            block(32, 64),
            block(64, 128),
            block(128, 256),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.head(x)
        return x


def load_class_names() -> list[str]:
    if not CLASS_NAMES_PATH.exists():
        st.error("class_names.json not found. Export it from the training notebook first.")
        st.stop()

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_model(model_type: str, num_classes: int) -> nn.Module:
    if model_type == "convnext":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT
        model = models.convnext_tiny(weights=weights)

        in_features = model.classifier[2].in_features

        model.classifier = nn.Sequential(
            model.classifier[0],
            model.classifier[1],
            nn.Dropout(0.1),
            nn.Linear(in_features, num_classes)
        )

        return model

    if model_type == "vit":
        from transformers import ViTConfig, ViTForImageClassification

        config = ViTConfig(
            image_size=224,
            patch_size=16,
            num_channels=3,
            num_labels=num_classes,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            qkv_bias=True,
        )

        model = ViTForImageClassification(config)

        return model

    if model_type == "cnn":
        return BaselineCNN(num_classes=num_classes, dropout=0.4)

    if model_type == "resnet":
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unknown model type: {model_type}")


@st.cache_resource
def load_model(model_name: str, num_classes: int) -> nn.Module:
    config = MODEL_CONFIG[model_name]
    model_path = config["path"]

    if not model_path.exists():
        st.error(f"Model file not found: {model_path}")
        st.stop()

    model = build_model(config["type"], num_classes)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def get_transform(model_type: str) -> transforms.Compose:
    if model_type == "cnn":
        img_size = 160
    elif model_type == "convnext":
        img_size = 256
    else:
        img_size = 224

    if model_type == "vit":
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def format_class_name(name: str) -> str:
    return name.replace("_", " ").title()


def animal_type(class_name: str) -> str:
    return "Dog" if class_name.lower() in DOG_CLASSES else "Cat"


def predict(
    image: Image.Image,
    model: nn.Module,
    class_names: list[str],
    model_type: str,
    top_k: int = 3,
) -> list[dict]:
    transform = get_transform(model_type)
    x = transform(image.convert("RGB")).unsqueeze(0)

    with torch.no_grad():
        if model_type == "vit":
            outputs = model(pixel_values=x)
            logits = outputs.logits
        else:
            logits = model(x)

        probs = F.softmax(logits, dim=1)[0]
        top_probs, top_indices = torch.topk(probs, k=top_k)

    results = []
    for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
        class_name = class_names[idx]
        results.append({
            "class_name": class_name,
            "display_name": format_class_name(class_name),
            "animal_type": animal_type(class_name),
            "confidence": prob,
        })

    return results


# -----------------------------
# UI
# -----------------------------
st.title("🐾 Pet Breed Classifier")
st.write("Upload a pet image, choose a model, and run inference to predict the breed.")

class_names = load_class_names()

with st.sidebar:
    st.header("Model settings")
    model_name = st.selectbox("Choose model", list(MODEL_CONFIG.keys()))
    st.caption("Recommended final model: ConvNeXt. Baseline CNN is included for comparison.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    run = st.button("Run", type="primary")

    if run:
        with st.spinner("Running model prediction..."):
            model = load_model(model_name, len(class_names))
            model_type = MODEL_CONFIG[model_name]["type"]
            results = predict(image, model, class_names, model_type=model_type, top_k=3)

        best = results[0]
        st.subheader("Prediction")
        st.success(f"{best['animal_type']} - {best['display_name']}")
        st.metric("Confidence", f"{best['confidence'] * 100:.2f}%")

        st.subheader("Top-3 predictions")
        for r in results:
            st.write(f"**{r['display_name']}** ({r['animal_type']}) - {r['confidence'] * 100:.2f}%")
            st.progress(float(r["confidence"]))
else:
    st.info("Upload an image to start.")
