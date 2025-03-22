import torch
import torch.nn as nn


class HandModel(nn.Module):
    def __init__(self, num_classes=6):
        super(HandModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 12, 64)  # 48 → 24 → 12
        self.fc2 = nn.Linear(64, num_classes)  # 27

        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.to(torch.float32).unsqueeze(1)  # (batch, 48) → (batch, 1, 48)
        x = self.relu(self.conv1(x))  # (batch, 16, 48)
        x = self.pool(x)  # (batch, 16, 24)
        x = self.relu(self.conv2(x))  # (batch, 32, 24)
        x = self.pool(x)  # (batch, 32, 12)

        x = x.view(x.size(0), -1)  # flatten (batch, 32 * 12)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    model = HandModel()
    print(model)

    # test
    test_input = torch.randn(1, 48)  # batch_size = 1, 48
    output = model(test_input)
    print("Output shape:", output.shape)  # (1, 27)
