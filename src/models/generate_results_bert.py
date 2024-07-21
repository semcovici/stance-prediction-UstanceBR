import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.nn.functional import softmax
from ast import literal_eval
from tqdm import tqdm
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModel
import random
from transformers import set_seed, TrainerCallback
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from belt_nlp.bert_with_pooling import BertClassifierWithPooling
from torch.nn.functional import softmax
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
import sys
import os
import csv
sys.path.append("src/")
from models.classification_methods import create_test_results_df
from data.lambdas import int_to_label, label_to_int

tqdm.pandas()
random.seed(0)
set_seed(42)

# Define paths
raw_data_path = 'data/raw/'
processed_data_path = 'data/processed/'
reports_path = 'reports/'
file_format_tmt = processed_data_path + "{split}_r3_{target}_top_mentioned_timelines_processed.csv"
file_format_users = processed_data_path + 'r3_{target}_{split}_users_processed.csv' 

# Target list
target_list = [
    'ig',
    'bo', 
    'cl', 
    'co', 
    'gl', 
    'lu'
]

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Definir o modelo pré-treinado
model_name = "pablocosta/bertabaporu-base-uncased"

# Função para tokenizar os dados
def tokenize_function(examples):
    
    if exp_name == "Stance":
        return tokenizer(examples['text'], padding='max_length', truncation=True)
    else:
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)    

# Verificar se a GPU está disponível e definir o dispositivo
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print(device)


dict_exps = {
    "Stance": {
        'path_dataset': file_format_users,
        "text_col": "Stance",
        "batch_size": 16,
        "epochs": 3,
        "pre_tokenize": False
    },
    "Timeline": {
        'path_dataset': file_format_users,
        "text_col": "Timeline",
        "batch_size": 2,
        "epochs": 3,
        "pre_tokenize": True
    },
    "Texts": {
        'path_dataset': file_format_tmt,
        "text_col": "Texts",
        "batch_size": 2,
        "epochs": 3,
        "pre_tokenize": True
    },
}

check_if_already_exists = True

def pre_tokenize(string, n_tokens):
    
    tokens = string.split(" ")
    
    if len(tokens) <= n_tokens:
        return string
    
    new_string = " ".join(tokens[:n_tokens])
    
    if new_string in string:
        return new_string
    else:
        raise Exception("erro")
    
class SaveMetricsCallback(TrainerCallback):
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "training_metrics.csv")
        with open(self.log_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["epoch", "step", "metric", "value"])

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            with open(self.log_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                for key, value in logs.items():
                    writer.writerow([state.epoch, state.global_step, key, value])



for exp_name, config in dict_exps.items():
    
    
    print(f"""###########################################
# Running: {exp_name} 
###########################################""")
    
    
    text_col = config['text_col']
    path_dataset = config['path_dataset']
    
    # Processar cada target
    for target in target_list:
        
        print(f"""######## target: {target}""")
        estimator_name = "bert_classifier_" + model_name.replace("/","_").replace("-","_")
        test_results_path = f"{reports_path}test_results/{estimator_name}_{target}_{text_col}_test_results.csv"
        train_results_path = f"{reports_path}train_results/{estimator_name}_{target}_{text_col}_train_results.csv"
        
        if os.path.isfile(test_results_path) and check_if_already_exists:
            print('# experiment already done')
            continue
        
        
        
        # Ler e dividir os dados
        train_val = pd.read_csv(
            path_dataset.format(target=target, split="train"), 
            sep=';', 
            encoding='utf-8-sig'
        ).reset_index()[[text_col, 'Polarity']].rename(columns={text_col: 'text', 'Polarity': 'label'})
        
        train_val.label = train_val.label.apply(lambda x: label_to_int(x))

        # check if label is binary
        if len(train_val.label.unique()) != 2:
            raise Exception("There is an error in train_val label transformation: expected to be binary")
        
        train, val = train_test_split(train_val, test_size=0.15, random_state=42)
        train.reset_index(drop=True, inplace=True)
        val.reset_index(drop=True, inplace=True)
        
        test = pd.read_csv(
            path_dataset.format(target=target, split="test"), 
            sep=';', 
            encoding='utf-8-sig'
        ).reset_index()[[text_col, 'Polarity']].rename(columns={text_col: 'text', 'Polarity': 'label'})
        
        test.label = test.label.apply(lambda x: label_to_int(x))

        # check if label is binary
        if len(test.label.unique()) != 2:
            raise Exception("There is an error in test label transformation: expected to be binary")
        
        # pre tokenizacao (tirar o grosso dos tokens antes de jogar no modelo) - util pra quando tem colunas com textos MUITO longos
        train.text = train.text.apply(lambda x: pre_tokenize(x, n_tokens = 2000))
        test.text = test.text.apply(lambda x: pre_tokenize(x, n_tokens = 2000))
        val.text = val.text.apply(lambda x: pre_tokenize(x, n_tokens = 2000))
         
         
        
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

        batch_size = config['batch_size']
        epochs = config['epochs']
        
        log_dir = f'./bert_training_logs/{exp_name}_{target}'

        # Definir os argumentos de treinamento
        training_args = TrainingArguments(
            output_dir=f'./bert_training_results/{exp_name}_{target}',
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            logging_dir=log_dir,
            logging_steps=10,
            load_best_model_at_end=True,
            report_to="all"
        )

        # Definir o Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['val'],
            tokenizer=tokenizer,
            callbacks=[SaveMetricsCallback(log_dir=log_dir)]
        )

        # Treinar o modelo
        trainer.train()

        # Avaliar o modelo no conjunto de teste
        eval_results = trainer.evaluate(eval_dataset=tokenized_datasets['test'])
        print(f"Evaluation results for {target}: {eval_results}")

        # Predição no conjunto de teste
        test_predictions = trainer.predict(test_dataset=tokenized_datasets['test'])
        test_pred_labels = np.argmax(test_predictions.predictions, axis=1)

        # Predição no conjunto de treino
        train_predictions = trainer.predict(test_dataset=tokenized_datasets['train'])
        train_pred_labels = np.argmax(train_predictions.predictions, axis=1)

        # get logits
        test_pred_logits = test_predictions.predictions
        train_pred_logits = train_predictions.predictions

        # transform logits in "probabilities"
        test_pred_probs = softmax(torch.tensor(test_pred_logits), dim=-1).numpy()
        train_pred_probs = softmax(torch.tensor(train_pred_logits), dim=-1).numpy()

        # create list of proba of each class
        test_proba_0 = [float(probas[0]) for probas in test_pred_probs]
        test_proba_1 = [float(probas[1]) for probas in test_pred_probs]
        train_proba_0 = [float(probas[0]) for probas in train_pred_probs]
        train_proba_1 = [float(probas[1]) for probas in train_pred_probs]

        # create list of test and prediction
        y_test = test['label'].tolist()
        y_train = train['label'].tolist()
        y_test_pred = test_pred_labels.tolist()
        y_train_pred = train_pred_labels.tolist()

        # format test and pred
        y_test_formated = [int_to_label(test) for test in y_test]
        y_train_formated = [int_to_label(train) for train in y_train]
        y_test_pred_formated = [int_to_label(pred) for pred in y_test_pred]
        y_train_pred_formated = [int_to_label(pred) for pred in y_train_pred]

        # create df with results
        df_test_results = create_test_results_df(y_test_formated, y_test_pred_formated, test_proba_0, test_proba_1)
        df_train_results = create_test_results_df(y_train_formated, y_train_pred_formated, train_proba_0, train_proba_1)

        df_test_results.to_csv(test_results_path, index=False)
        df_train_results.to_csv(train_results_path, index=False)