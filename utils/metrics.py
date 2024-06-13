
################ RECOMMENDATION

#import pandas as pd
#from sklearn.metrics.pairwise import cosine_similarity
#from sklearn.feature_extraction.text import TfidfVectorizer
#
#def calculate_similarity(content, true_value):
#    vectorizer = TfidfVectorizer().fit_transform([content, true_value])
#    vectors = vectorizer.toarray()
#    return cosine_similarity(vectors)[0, 1]
#
#def calculate_average_similarity(df):
#    df['similarity'] = df.apply(lambda row: calculate_similarity(row['content'], row['true_value']), axis=1)
#    average_similarity = df['similarity'].mean()
#    return average_similarity
#
#def generate_recommendation_metrics(df, bpm_task='activity_recommendation'):
#    task_df = df[df['bpm_task'] == bpm_task]
#    grouped = task_df.groupby(['model_name', 'prompt_pattern'])
#    
#    results = []
#    for (model_name, prompt_pattern), group in grouped:
#        average_similarity = calculate_average_similarity(group)
#        results.append({
#            'model_name': model_name,
#            'prompt_pattern': prompt_pattern,
#            'average_similarity': average_similarity
#        })
#    
#    result_df = pd.DataFrame(results)
#    return result_df

################ RECOMMENDATION
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def embed_sentence(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    # Use the embeddings from the [CLS] token
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding.detach().numpy()

def calculate_similarity(content, true_value):
    content_embedding = embed_sentence(content)
    true_value_embedding = embed_sentence(true_value)
    return cosine_similarity(content_embedding, true_value_embedding)[0, 0]

def calculate_average_similarity(df):
    df['similarity'] = df.apply(lambda row: calculate_similarity(row['content'], row['true_value']), axis=1)
    average_similarity = df['similarity'].mean()
    return average_similarity

def generate_recommendation_metrics(df, bpm_task='activity_recommendation'):
    task_df = df[df['bpm_task'] == bpm_task]
    grouped = task_df.groupby(['model_name', 'prompt_pattern'])
    
    results = []
    for (model_name, prompt_pattern), group in grouped:
        average_similarity = calculate_average_similarity(group)
        results.append({
            'model_name': model_name,
            'prompt_pattern': prompt_pattern,
            'average_similarity': average_similarity
        })
    
    result_df = pd.DataFrame(results)
    return result_df

################ RPA

import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

def calculate_failure_rate(group):
    failures = group['content'].apply(lambda x: not x.isdigit()).sum()
    total = len(group)
    failure_rate = failures / total
    return failure_rate
def calculate_precision_recall_f1(group):
    y_true = group['true_value'].apply(str)
    y_pred = group['content'].apply(lambda x: x if x.isdigit() else '-1')  # invalid predictions as -1
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=['0', '1', '2'], average=None)
    overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=['0', '1', '2'], average='weighted')
    
    metrics = {
        'precision_0': precision[0],
        'recall_0': recall[0],
        'f1_0': f1[0],
        'precision_1': precision[1],
        'recall_1': recall[1],
        'f1_1': f1[1],
        'precision_2': precision[2],
        'recall_2': recall[2],
        'f1_2': f1[2],
        'overall_precision': overall_precision,
        'overall_recall': overall_recall,
        'overall_f1': overall_f1
    }
    return metrics
def generate_rpa_metrics(df, bpm_task='rpa'):
    results = []
    df = df[df['bpm_task'] == bpm_task]
    grouped = df.groupby(['model_name', 'prompt_pattern'])
    
    for name, group in grouped:
        model_name, prompt_pattern = name
        failure_rate = calculate_failure_rate(group)
        metrics = calculate_precision_recall_f1(group)
        result = {
            'model_name': model_name,
            'prompt_pattern': prompt_pattern,
            'failure_rate': failure_rate,
            **metrics
        }
        results.append(result)
    
    return pd.DataFrame(results)


################## CONSTRAINTS

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

def generate_constraints_metrics(dataframe):
    import warnings
    warnings.filterwarnings('ignore')
    # Filter for constraints task
    df_constraints = dataframe[dataframe['bpm_task'] == 'constraints']
    
    # Define possible constraint types
    constraint_types = ['precedence', 'response', 'succession', 'init', 'end']
    
    # Function to calculate precision, recall, and F1 for each constraint type
    def calculate_metrics(true_values, predicted_values, constraint_type):
        y_true = [constraint_type in true_val.split(',') for true_val in true_values]
        y_pred = [constraint_type in pred_val.split(',') for pred_val in predicted_values]
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        return precision, recall, f1
    
    # Group by model_name and prompt_pattern
    grouped = df_constraints.groupby(['model_name', 'prompt_pattern'])
    
    results = []
    
    for (model_name, prompt_pattern), group in grouped:
        true_values = group['true_value'].str.lower()
        predicted_values = group['content'].str.lower()
        
        metrics = {'model_name': model_name, 'prompt_pattern': prompt_pattern}
        
        # Calculate metrics for each constraint type
        for constraint_type in constraint_types:
            precision, recall, f1 = calculate_metrics(true_values, predicted_values, constraint_type)
            metrics[f'{constraint_type}_precision'] = precision
            metrics[f'{constraint_type}_recall'] = recall
            metrics[f'{constraint_type}_f1'] = f1
        
        # Calculate overall metrics
        y_true = [ct in tv.split(',') for tv in true_values for ct in constraint_types]
        y_pred = [ct in pv.split(',') for pv in predicted_values for ct in constraint_types]
        overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        
        metrics['overall_precision'] = overall_precision
        metrics['overall_recall'] = overall_recall
        metrics['overall_f1'] = overall_f1
        
        results.append(metrics)
    
    return pd.DataFrame(results)


########## PROCESS QA

import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import numpy as np

def generate_qa_metrics(data):
    # Initialize the ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    
    # Group data by bpm_task, model_name, and prompt_pattern
    grouped = data.groupby(['bpm_task', 'model_name', 'prompt_pattern'])
    
    results = []

    for (bpm_task, model_name, prompt_pattern), group in grouped:
        if bpm_task == 'process_qa':
            # Calculate BLEU and ROUGE scores for different complexity levels
            complexities = ['easy', 'medium', 'complex']
            for complexity in complexities:
                subset = group[group['note'].str.contains(complexity)]
                if not subset.empty:
                    bleu_scores = [sentence_bleu([true.split()], pred.split()) for true, pred in zip(subset['true_value'], subset['content'])]
                    rouge_scores = [scorer.score(true, pred) for true, pred in zip(subset['true_value'], subset['content'])]
                    
                    avg_bleu = np.mean(bleu_scores)
                    avg_rouge = {key: np.mean([score[key].fmeasure for score in rouge_scores]) for key in rouge_scores[0]}

                    results.append({
                        'bpm_task': bpm_task,
                        'model_name': model_name,
                        'prompt_pattern': prompt_pattern,
                        'complexity': complexity,
                        'avg_bleu': avg_bleu,
                        'avg_rouge1': avg_rouge['rouge1'],
                        'avg_rougeL': avg_rouge['rougeL']
                    })
            
            # Calculate overall BLEU and ROUGE scores across all complexity levels
            bleu_scores = [sentence_bleu([true.split()], pred.split()) for true, pred in zip(group['true_value'], group['content'])]
            rouge_scores = [scorer.score(true, pred) for true, pred in zip(group['true_value'], group['content'])]

            avg_bleu = np.mean(bleu_scores)
            avg_rouge = {key: np.mean([score[key].fmeasure for score in rouge_scores]) for key in rouge_scores[0]}

            results.append({
                'bpm_task': bpm_task,
                'model_name': model_name,
                'prompt_pattern': prompt_pattern,
                'complexity': 'overall',
                'avg_bleu': avg_bleu,
                'avg_rouge1': avg_rouge['rouge1'],
                'avg_rougeL': avg_rouge['rougeL']
            })
    
    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)
    return results_df