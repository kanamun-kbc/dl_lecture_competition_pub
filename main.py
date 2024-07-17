import numpy as np
import torch
from train import train_model, evaluate_model
from src.datasets import VQADataset
from src.preprocs import get_image_transform, preprocess_question, tokenizer
from src.models.base import VQAModel
from torch.utils.data import DataLoader
import datetime

def main():
    data_dir = '/workspace/dl_lecture_competition_pub/data'
    batch_size = 32
    epochs = 1
    lr = 0.001

    device = torch.device('cuda')

    transform = get_image_transform()
    train_dataset = VQADataset(data_dir=data_dir, split='train', transform=transform, preprocess_fn=preprocess_question)
    valid_dataset = VQADataset(data_dir=data_dir, split='valid', transform=transform, preprocess_fn=preprocess_question)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # モデルの訓練と評価
    print("Starting training...")
    model, all_answers = train_model(data_dir, batch_size, epochs, lr, valid_dataloader)
    
    # モデルの最終評価
    print("Starting final evaluation...")
    predictions, accuracy = evaluate_model(model, valid_dataloader, all_answers)
    
    # タイムスタンプを使用して submission.npy を保存
    now = datetime.datetime.now()
    np.save("submission_" + now.strftime('%Y%m%d_%H%M%S') + ".npy", predictions)
    print(f"Final Evaluation complete. Submission file saved with accuracy: {accuracy:.4f}")

if __name__ == '__main__':
    main()
