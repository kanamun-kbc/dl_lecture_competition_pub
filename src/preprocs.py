import nltk
from nltk.corpus import stopwords
from transformers import BertTokenizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def preprocess_question(question):
    # 大文字・小文字の統一
    question = question.lower()

    # 冠詞の削除
    words = nltk.word_tokenize(question)
    words = [word for word in words if word not in ['a', 'an', 'the']]
    question = ' '.join(words)
    
    return question

def get_image_transform():
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform
