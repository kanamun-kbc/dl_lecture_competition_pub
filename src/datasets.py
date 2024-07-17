import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import json

class VQADataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None, preprocess_fn=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.preprocess_fn = preprocess_fn
        self.images = []
        self.questions = []
        self.answers = []
        self._load_data()
        self._print_stats()

    def _load_data(self):
        json_file = os.path.join(self.data_dir, f'{self.split}.json')
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"JSON file not found: {json_file}")

        with open(json_file, 'r') as f:
            data = json.load(f)

        image_dict = data.get('image', {})
        question_dict = data.get('question', {})
        answer_dict = data.get('answers', {})

        for key in image_dict:
            image_path = os.path.join(self.data_dir, self.split, image_dict[key])
            if os.path.exists(image_path):
                self.images.append(image_path)
                question = question_dict.get(key, "")
                answers = answer_dict.get(key, [])

                if self.preprocess_fn:
                    question = self.preprocess_fn(question)

                # 最も確信度の高い解答を使用
                if answers:
                    answer = max(answers, key=lambda x: x['answer_confidence'])['answer']
                else:
                    answer = "unknown"  # 解答がない場合のデフォルト値

                self.questions.append(question)
                self.answers.append(answer)

                # デバッグ用に質問とその解答を出力
                print(f"Question: {question}, Answer: {answer}")
            else:
                print(f"Image file not found: {image_path}")

        print(f"Loaded {len(self.images)} images, {len(self.questions)} questions, and {len(self.answers)} answers.")

    def _print_stats(self):
        unique_answers = set(self.answers)
        print(f"Number of unique answers in {self.split} set: {len(unique_answers)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        question = self.questions[idx]
        answer = self.answers[idx]

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, question, answer
