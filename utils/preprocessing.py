import os
import random
import pandas as pd
import pm4py

def get_random_xes_files(folder_path, num_files=5, seed=50):
    random.seed(seed)
    xes_files = [file for file in os.listdir(folder_path) if file.endswith('.xes')]
    random_files = random.sample(xes_files, num_files)
    return [os.path.join(folder_path, file) for file in random_files]

def read_event_log(file_path):
    return pm4py.read_xes(file_path)

def get_random_processes_from_logs(event_logs, num_processes=100, sequence_length=4, seed=50):
    random.seed(seed)
    all_traces = pd.concat(event_logs, ignore_index=True)
    trace_counts = all_traces['case:concept:name'].value_counts()
    valid_trace_ids = trace_counts[trace_counts >= sequence_length].index
    if len(valid_trace_ids) < num_processes:
        raise ValueError(f"Not enough valid traces in the logs. Found {len(valid_trace_ids)}, but need {num_processes}.")
    sampled_trace_ids = random.sample(list(valid_trace_ids), num_processes)
    sampled_traces = all_traces[all_traces['case:concept:name'].isin(sampled_trace_ids)]
    return sampled_traces


def get_random_sequences(trace, num_sequences, sequence_length=4):
    sequences = []
    if len(trace) >= sequence_length:
        start_indices = random.sample(range(len(trace) - sequence_length + 1), num_sequences)
        for start_idx in start_indices:
            seq = trace[start_idx:start_idx + sequence_length]
            sequences.append(seq)
    return sequences

def create_samples(traces, num_sequences_per_trace, sequence_length=4):
    samples = []
    trace_ids = traces['case:concept:name'].unique()
    for trace_id in trace_ids:
        trace = traces[traces['case:concept:name'] == trace_id].reset_index(drop=True)
        sequences = get_random_sequences(trace, num_sequences_per_trace, sequence_length)
        for seq in sequences:
            task = ", ".join(seq['concept:name'].iloc[:-1])
            manual_label = seq['concept:name'].iloc[-1]
            samples.append({"task": task, "manual_labels": manual_label})
    return samples


def main_preprocessing_activity_recommendation(folder_path, num_files=5, num_processes=5, num_sequences_per_trace=1, sequence_length=4, save=False):
    random_xes_files = get_random_xes_files(folder_path, num_files)
    event_logs = [read_event_log(file_path) for file_path in random_xes_files]
    random_processes = get_random_processes_from_logs(event_logs, num_processes, sequence_length)
    samples = create_samples(random_processes, num_sequences_per_trace, sequence_length)
    df = pd.DataFrame(samples)
    df = df.drop_duplicates().reset_index(drop='index')
    if save:

        df.to_excel('data/bpm_activity_recommendataion.xlsx', index= False)
    return df