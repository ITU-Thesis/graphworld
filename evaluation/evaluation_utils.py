import json
import pandas as pd
import itertools

def read_processed_shards(PROCESSED_DIR, shard=None):
    with open(f'{PROCESSED_DIR}/summary.json', 'r') as f:
        summary = json.load(f)

    dfs = []
    file_names = []
    if shard == None:
        for shard_idx in range(summary['N_RUNS']):
            filename = f'{shard_idx+1}.ndjson'
            file_names.append(filename)
    else:
        file_names.append(f'{shard}.ndjson')

    for f in file_names:
        print(f)
        with open(f'{PROCESSED_DIR}/shards/{f}', 'r') as f:
            lines = f.readlines()
            records = map(json.loads, lines)
            dfs.append(pd.DataFrame.from_records(records))

    # Construct df
    print("concatenating")
    results_df = pd.concat(dfs)
    del dfs
    return results_df.reset_index(drop=True)

def get_best_configuration_per_model(df, TEST_METRIC, n_best=1):
    best_configurations = {}

    # Common hparams
    COMMON_PARAMS = ['encoder_in_channels', 'encoder_hidden_channels', 'encoder_num_layers', 
                     'encoder_dropout', 'train_downstream_lr', 'train_downstream_epochs', 'train_patience']

    # GAT hparams
    GAT_PARAMS =['encoder_heads']

    # JL hparams
    JL_PARAMS = COMMON_PARAMS + ['train_pretext_weight']

    # URL & PF hparams
    TWO_STAGE_PARAMS = COMMON_PARAMS + ['train_pretext_epochs', 'train_pretext_lr']

    models = [col.removesuffix(f'_{TEST_METRIC}') for col in df.columns if TEST_METRIC in col]
    for model in models:
        if "JL" in model:
            params = [f'{model}_{p}' for p in JL_PARAMS]
        else:
            params = [f'{model}_{p}' for p in TWO_STAGE_PARAMS]
        if "GAT" in model:
            params += [f'{model}_{p}' for p in GAT_PARAMS]
        test_metric = f'{model}_{TEST_METRIC}'
        df_m = df[params + [test_metric]]
        groups = df_m.groupby(params)
        means = groups.mean()
        if n_best == 0:
            n_best = groups.ngroups
        best_configuration = means[test_metric].nlargest(n_best).reset_index()
        best_configurations[model] = best_configuration
    return best_configurations


def unpivot_ssl_model(df : pd.DataFrame, suffix : str, ssl_models, encoders, training_schemes):
    '''
    Unpivot the results to a long format for all SSL methods. Each row corresponds to an experiment on graph.
    '''
    frames = []
    for (ssl_model, encoder, scheme) in itertools.product(*[ssl_models, encoders, training_schemes]):
        column = f'{encoder}_{ssl_model}_{scheme}_{suffix}'
        pretext_weight_col = f'{encoder}_{ssl_model}_{scheme}_train_pretext_weight'
        if not column in df.columns:
            continue
        df_model = df[[column]].rename(columns=lambda col: col.replace(column, suffix))
        df_model['pretext_weight'] = df[pretext_weight_col] if pretext_weight_col in df.columns else None    
        df_model['SSL_model'] = ssl_model
        df_model['Encoder'] = encoder
        df_model['Training_scheme'] = scheme
        df_model['Graph_ID'] = df.index.values.tolist()

        frames += [df_model]
    return pd.concat(frames, ignore_index=True)

def unpivot_bvaseline_model(df : pd.DataFrame, suffix : str, baseline_models, training_schemes):
    '''
    Unpivot the results to a long format for all baseline methods. Each row corresponds to an experiment on graph.
    '''
    frames = []
    for baseline_model, training_scheme in itertools.product(*[baseline_models, training_schemes]):
        column = f'{baseline_model}__{training_scheme}_{suffix}'
        if not column in df.columns:
            continue
        df_model = df[[column]].rename(columns=lambda col: col.replace(column, suffix))
        df_model['Baseline_model'] = baseline_model
        df_model['Graph_ID'] = df.index.values.tolist()
        
        frames += [df_model]
    return pd.concat(frames, ignore_index=True)
