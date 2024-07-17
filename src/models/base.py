import torch
import torch.nn as nn
import torchvision.models as models
from transformers import BertModel

class VQAModel(nn.Module):
    def __init__(self, num_classes):
        super(VQAModel, self).__init__()
        # ResNet50モデルを最新のweightsパラメータを使用してロード
        self.image_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # 元の全結合層の in_features を保存
        self.image_model_in_features = self.image_model.fc.in_features
        self.image_model.fc = nn.Identity()  # 最後の全結合層を除去

        self.text_model = BertModel.from_pretrained('bert-base-uncased')

        self.fc = nn.Linear(self.image_model_in_features + self.text_model.config.hidden_size, 512)
        self.classifier = nn.Linear(512, num_classes)  # 回答クラス数に基づいて設定

    def forward(self, image, input_ids, attention_mask):
        image_features = self.image_model(image)
        text_features = self.text_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]

        combined_features = torch.cat((image_features, text_features), dim=1)
        x = self.fc(combined_features)
        x = self.classifier(x)

        return x
