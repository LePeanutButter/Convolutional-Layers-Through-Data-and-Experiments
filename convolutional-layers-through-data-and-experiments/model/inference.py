import json
import torch
import torch.nn as nn

class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(self.relu(self.conv(x)))


class CNNClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            ConvolutionalBlock(1, 32),
            ConvolutionalBlock(32, 64),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def model_fn(model_dir):
    model = CNNClassifier()
    model.load_state_dict(
        torch.load(f"{model_dir}/model.pth", map_location="cpu")
    )
    model.eval()
    return model


def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        data = json.loads(request_body)
        tensor = torch.tensor(data["inputs"], dtype=torch.float32)
        return tensor
    raise ValueError("Unsupported content type")


def predict_fn(input_data, model):
    with torch.no_grad():
        outputs = model(input_data)
        predictions = torch.argmax(outputs, dim=1)
    return predictions


def output_fn(prediction, content_type):
    return json.dumps({"prediction": prediction.tolist()})
