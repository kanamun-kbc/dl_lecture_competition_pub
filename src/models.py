import torch
import torch.nn as nn

class WaveNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, dilation=1):
        super(WaveNetBlock, self).__init__()
        self.conv_filter = nn.Conv1d(in_channels, out_channels, kernel_size, padding=dilation, dilation=dilation)
        self.conv_gate = nn.Conv1d(in_channels, out_channels, kernel_size, padding=dilation, dilation=dilation)
        self.conv_skip = nn.Conv1d(out_channels, out_channels, 1)
        self.conv_residual = nn.Conv1d(out_channels, out_channels, 1)

    def forward(self, x):
        filter_output = torch.tanh(self.conv_filter(x))
        gate_output = torch.sigmoid(self.conv_gate(x))
        output = filter_output * gate_output
        skip_output = self.conv_skip(output)
        residual_output = self.conv_residual(output) + x
        return skip_output, residual_output

class WaveNet(nn.Module):
    def __init__(self, num_classes, num_channels=128, residual_channels=64, skip_channels=128, num_blocks=4, num_layers=10):
        super(WaveNet, self).__init__()
        self.input_conv = nn.Conv1d(num_channels, residual_channels, 1)
        self.blocks = nn.ModuleList()
        for b in range(num_blocks):
            for l in range(num_layers):
                dilation = 2 ** l
                self.blocks.append(WaveNetBlock(residual_channels, skip_channels, dilation=dilation))
        self.output_conv1 = nn.Conv1d(skip_channels, skip_channels, 1)
        self.output_conv2 = nn.Conv1d(skip_channels, num_classes, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.input_conv(x)
        skip_connections = []
        for block in self.blocks:
            skip, x = block(x)
            skip_connections.append(skip)
        x = sum(skip_connections)
        x = torch.relu(self.output_conv1(x))
        x = self.dropout(x)
        x = self.output_conv2(x)
        x = torch.mean(x, dim=2)
        return x

class SubjectSpecificLayer(nn.Module):
    def __init__(self, num_subjects, feature_size):
        super(SubjectSpecificLayer, self).__init__()
        self.embeddings = nn.Embedding(num_subjects, feature_size)

    def forward(self, x, subject_idx):
        subject_features = self.embeddings(subject_idx)
        return x + subject_features.unsqueeze(2)

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

class EnhancedWaveNet(nn.Module):
    def __init__(self, num_classes, num_subjects, num_channels, hid_dim=64, p_drop=0.5):
        super(EnhancedWaveNet, self).__init__()
        self.input_conv = nn.Conv1d(num_channels, hid_dim, kernel_size=1)
        self.hidden_conv = nn.Conv1d(hid_dim, hid_dim, kernel_size=1)
        self.dropout = nn.Dropout(p_drop)
        self.fc = nn.Linear(hid_dim, num_classes)
    
    def forward(self, x, subject_idx):
        x = self.input_conv(x)
        x = F.relu(x)
        x = self.hidden_conv(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = torch.mean(x, dim=-1)
        x = self.fc(x)
        return x

