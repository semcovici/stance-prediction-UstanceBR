import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.nn.functional import softmax
from ast import literal_eval
from tqdm import tqdm
from sklearn.metrics import classification_report
tqdm.pandas()
from transformers import AutoTokenizer, AutoModel
import random
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
random.seed(0)
from belt_nlp.bert_with_pooling import BertClassifierWithPooling
from torch.nn.functional import softmax
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
import sys
import os
sys.path.append("src/")
from models.classification_methods import create_test_results_df
from data.lambdas import int_to_label, label_to_int


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
    
    if text_col == 'Stance':
        # resultados de stance estao com esse, nao alterar antes de atualizar
        return tokenizer(examples['text'], padding='max_length', max_length = 512)
    else:
        return tokenizer(examples['text'], padding='max_length' , max_length = 512, truncation = True)
    
# Verificar se a GPU está disponível e definir o dispositivo
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print(device)


dict_exps = {
    "Stance": {
        'path_dataset': file_format_users,
        "text_col": "Stance",
    },
    "Timeline": {
        'path_dataset': file_format_users,
        "text_col": "Timeline",
    },
    "Texts": {
        'path_dataset': file_format_tmt,
        "text_col": "Texts",
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

        # get logits
        test_pred_logits = test_predictions.predictions
        # transform logits in "probabilities"
        test_pred_probs = softmax(torch.tensor(test_pred_logits), dim=-1).numpy()

        # create list of proba of each class
        pred_proba_0 = [float(probas[0]) for probas in test_pred_probs]
        pred_proba_1 = [float(probas[1]) for probas in test_pred_probs]

        # create list of test and prediction
        y_test = test['label'].tolist()
        y_pred = test_pred_labels.tolist()

        # format test and pred
        y_test_formated = [int_to_label(test) for test in y_test]
        y_pred_formated = [int_to_label(pred) for pred in y_pred]

        # create df with results
        df_test_results = create_test_results_df(y_test_formated, y_pred_formated, pred_proba_0, pred_proba_1)
        
        df_test_results.to_csv(test_results_path, index=False)