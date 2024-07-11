import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from ast import literal_eval
from tqdm import tqdm
from sklearn.metrics import classification_report
tqdm.pandas()
from transformers import AutoTokenizer, AutoModel
import random
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
random.seed(0)
from belt_nlp.bert_with_pooling import BertClassifierWithPooling

# Define paths
raw_data_path = '../data/raw/'
processed_data_path = '../data/processed/'
reports_path = '../reports/'
file_format_users_filtered = processed_data_path + 'r3_{target}_{split}_users_scored_Timeline.csv' 
file_format_tmt_filtered = processed_data_path + '{split}_r3_{target}_top_mentioned_timelines_scored_Texts.csv'

# Target list
target_list = [
    'ig',
    'bo', 
    'cl', 
    'co', 
    'gl', 
    'lu'
]

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict

# Definir o modelo pré-treinado
model_name = "pablocosta/bertabaporu-base-uncased"

# Função para tokenizar os dados
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

# Verificar se a GPU está disponível e definir o dispositivo
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Processar cada target
for target in target_list:
    print(f"""
##########################
# Running: {target}
##########################          
          """)
    
    # Ler e dividir os dados
    train_val = pd.read_csv(
        file_format_users_filtered.format(target=target, split="train"), 
        sep=';', 
        encoding='utf-8-sig'
    ).reset_index()[['Stance', 'Polarity']].rename(columns={'Stance': 'text', 'Polarity': 'label'})
    
    train_val.label = train_val.label.map({"against": 0, "for": 1})
    
    train, val = train_test_split(train_val, test_size=0.15, random_state=42)
    train.reset_index(drop=True, inplace=True)
    val.reset_index(drop=True, inplace=True)
    
    test = pd.read_csv(
        file_format_users_filtered.format(target=target, split="test"), 
        sep=';', 
        encoding='utf-8-sig'
    ).reset_index()[['Stance', 'Polarity']].rename(columns={'Stance': 'text', 'Polarity': 'label'})
    
    test.label = test.label.map({"against": 0, "for": 1})

    
    # Criar datasets do Hugging Face
    train_dataset = Dataset.from_pandas(train)
    val_dataset = Dataset.from_pandas(val)
    test_dataset = Dataset.from_pandas(test)
    
    # Tokenizar os datasets
    tokenizer = BertTokenizer.from_pretrained(model_name)
    tokenized_datasets = DatasetDict({
        'train': train_dataset.map(tokenize_function, batched=True),
        'val': val_dataset.map(tokenize_function, batched=True),
        'test': test_dataset.map(tokenize_function, batched=True)
    })

    # Carregar o modelo e mover para o dispositivo (GPU se disponível)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

    # Definir os argumentos de treinamento
    training_args = TrainingArguments(
        output_dir=f'./results/{target}',
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir=f'./logs/{target}',
        logging_steps=10,
    )

    # Definir o Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['val'],
        tokenizer=tokenizer,
    )

    # Treinar o modelo
    trainer.train()

    # Avaliar o modelo no conjunto de teste
    eval_results = trainer.evaluate(eval_dataset=tokenized_datasets['test'])
    print(f"Evaluation results for {target}: {eval_results}")

    # Predição no conjunto de teste
    test_predictions = trainer.predict(test_dataset=tokenized_datasets['test'])
    test_pred_labels = np.argmax(test_predictions.predictions, axis=1)

    # Imprimir o classification report
    print(f"Classification Report for {target}:\n")
    print(classification_report(test['label'], test_pred_labels, target_names=['against', 'for']))
