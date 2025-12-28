import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import gradio as gr

# ----------------------------
# Model (same as your training)
# ----------------------------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class DenseNetViT(nn.Module):
    def __init__(self, num_classes=3, d_model=768, nhead=12, num_layers=6,
                 dim_feedforward=3072, dropout=0.1):
        super().__init__()
        backbone = models.densenet121(weights=None)
        self.backbone_features = backbone.features
        num_backbone_channels = backbone.classifier.in_features  # 1024

        self.token_proj = nn.Linear(num_backbone_channels, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        self.pos_encoding = SinusoidalPositionalEncoding(d_model)
        self.pos_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        feats = self.backbone_features(x)
        feats = F.relu(feats, inplace=True)

        B, C, Hf, Wf = feats.shape
        tokens = feats.flatten(2).transpose(1, 2)  # (B, Hf*Wf, C)

        tokens = self.token_proj(tokens)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, tokens), dim=1)

        x = self.pos_encoding(x)
        x = self.pos_dropout(x)

        x = self.transformer(x)
        cls_out = self.norm(x[:, 0])

        return self.head(cls_out)


# ----------------------------
# Config
# ----------------------------
CLASS_NAMES = ["Brain_Glioma", "Brain_Menin", "Brain_Tumor"]  # make sure order matches training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# ----------------------------
# Load model weights
# ----------------------------
model = DenseNetViT(num_classes=len(CLASS_NAMES)).to(DEVICE)
state = torch.load("best_model.pth", map_location=DEVICE)
model.load_state_dict(state)
model.eval()


@torch.no_grad()
def predict(image: Image.Image):
    if image is None:
        return {"error": "Please upload an image."}

    image = image.convert("RGB")
    x = transform(image).unsqueeze(0).to(DEVICE)

    logits = model(x)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    return {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Brain MRI Image"),
    outputs=gr.Label(num_top_classes=3, label="Prediction (probabilities)"),
    title="Multiclass Brain Tumor Classification",
    description=(
        "Upload an MRI image to classify into: Brain_Glioma, Brain_Menin, Brain_Tumor.\n\n"
    )
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)

