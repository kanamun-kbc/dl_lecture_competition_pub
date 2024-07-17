import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from src.datasets import VQADataset
from src.models.base import VQAModel
from src.preprocs import get_image_transform, preprocess_question, tokenizer
import datetime

def map_answer(answer, all_answers, reverse=False):
    if reverse:
        return all_answers[answer]
    if answer in all_answers:
        return all_answers[answer]
    else:
        new_index = len(all_answers)
        all_answers[answer] = new_index
        return new_index

def evaluate_model(model, dataloader, all_answers):
    model.eval()
    predictions = []
    correct = 0
    total = 0
    reverse_all_answers = {v: k for k, v in all_answers.items()}

    with torch.no_grad():
        for batch in dataloader:
            images, questions, answers = batch
            images = images.cuda()
            processed_questions = [preprocess_question(q) for q in questions]
            inputs = tokenizer(processed_questions, padding='max_length', truncation=True, max_length=30, return_tensors='pt')
            input_ids = inputs['input_ids'].squeeze(1).cuda()
            attention_mask = inputs['attention_mask'].squeeze(1).cuda()

            # 回答のマッピング
            mapped_answers = [all_answers[a] if a in all_answers else len(all_answers) for a in answers]
            answers_tensor = torch.tensor(mapped_answers).cuda()

            outputs = model(images, input_ids, attention_mask)
            _, preds = torch.max(outputs, 1)
            predictions.extend([reverse_all_answers[p.item()] for p in preds.cpu().numpy()])

            total += answers_tensor.size(0)
            correct += (preds == answers_tensor).sum().item()

    accuracy = correct / total
    return predictions, accuracy

def train_model(data_dir, batch_size, epochs, lr, valid_dataloader):
    # データセットとデータローダーの作成
    transform = get_image_transform()
    train_dataset = VQADataset(data_dir=data_dir, split='train', transform=transform, preprocess_fn=preprocess_question)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # すべての回答を収集
    all_answers = {}
    for _, _, answer in train_dataset:
        if answer not in all_answers:
            all_answers[answer] = len(all_answers)

    # 回答クラス数を取得
    num_classes = len(all_answers)
    print(f'Number of answer classes: {num_classes}')

    if num_classes <= 1:
        raise ValueError("The number of answer classes should be greater than 1.")

    # モデルの初期化
    model = VQAModel(num_classes=num_classes).cuda()
    model.train()

    # 損失関数とオプティマイザの設定
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_accuracy = 0

    # 訓練ループ
    for epoch in range(epochs):
        for batch in train_dataloader:
            images, questions, answers = batch
            optimizer.zero_grad()

            # 質問文の前処理とトークナイズ
            processed_questions = [preprocess_question(q) for q in questions]
            inputs = tokenizer(processed_questions, padding='max_length', truncation=True, max_length=30, return_tensors='pt')
            input_ids = inputs['input_ids'].squeeze(1).cuda()
            attention_mask = inputs['attention_mask'].squeeze(1).cuda()

            # 回答のマッピング
            mapped_answers = [map_answer(a, all_answers) for a in answers]
            answer_tensor = torch.tensor(mapped_answers).cuda()

            # モデルのフォワードパス
            outputs = model(images.cuda(), input_ids, attention_mask)
            loss = criterion(outputs, answer_tensor)

            # バックプロパゲーションと最適化
            loss.backward()
            optimizer.step()

        # 各エポック終了時に評価を実施
        model.eval()
        predictions, accuracy = evaluate_model(model, valid_dataloader, all_answers)
        model.train()

        # エポックごとの最良モデル保存
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            now = datetime.datetime.now()
            model_save_path = f"best_model_epoch_{epoch+1}_" + now.strftime('%Y%m%d_%H%M%S') + ".pth"
            torch.save(model.state_dict(), model_save_path)

        print(f'Epoch {epoch+1}, Loss: {loss.item()}, Validation Accuracy: {accuracy:.4f}')

    return model, all_answers
