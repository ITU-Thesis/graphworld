import json
import pandas as pd
import itertools
from constants import AUXILIARY_ALL, HYBRID_ALL, CONTRAST_ALL, GENERATION_ALL, TEST_METRIC,\
    GENERATION_BASED_CATEGORY, HYBRID_CATEGORY, CONTRAST_BASED, AUXILIARY_CATEGORY,\
    SSL_MODELS, ENCODERS, TRAINING_SCHEMES, BASELINES

def ssl_method_to_category(method):
    if method in AUXILIARY_ALL:
        return AUXILIARY_CATEGORY
    elif method in HYBRID_ALL:
        return HYBRID_CATEGORY
    elif method in CONTRAST_ALL:
        return CONTRAST_BASED
    elif method in GENERATION_ALL:
        return GENERATION_BASED_CATEGORY
    else:
        raise Exception('Unknown SSL method ' + method)

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
    for (ssl_model, encoder, scheme) in itertools.product(*[SSL_MODELS, ENCODERS, TRAINING_SCHEMES]):
        column = f'{encoder}_{ssl_model}_{scheme}_{TEST_METRIC}'
        if not column in results_df.columns:
            continue
        results_df[column] *= 100

    for baseline_model, training_scheme in itertools.product(*[BASELINES, TRAINING_SCHEMES]):
        column = f'{baseline_model}__{training_scheme}_{TEST_METRIC}'
        if not column in results_df.columns:
            continue
        results_df[column] *= 100
    
    del dfs
    return results_df.reset_index(drop=True)

def read_global_results(*args, **kwargs) -> pd.DataFrame:
    '''
    Read the global results having no marginalization
    '''
    df = read_processed_shards(*args, **kwargs)
    df.drop(['marginal_param', 'fixed_params'], axis=1, inplace=True)
    df.dropna(axis=0, inplace=True)
    return df

def read_multiple_global_results(experiments, **kwargs):
    dfs = []
    for (label, dir) in experiments:
        df = read_global_results(dir, **kwargs)
        df['Experiment'] = label
        dfs += [df]
    return pd.concat(dfs, ignore_index=True)


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
            n = groups.ngroups
        else:
            n = n_best
        best_configuration = means[test_metric].nlargest(n).reset_index()
        best_configurations[model] = best_configuration
    return best_configurations

def unpivot_ssl_model(df : pd.DataFrame, suffix : str, ssl_models, encoders, training_schemes, 
                      include_tuning_metric=False, include_graph_params=False, include_embeddings=False):
    '''
    Unpivot the results to a long format for all SSL methods. Each row corresponds to an experiment on graph.
    '''
    BENCHMARK_PARAMS = ['train_downstream_lr', 'train_pretext_weight',
               'train_pretext_epochs', 'train_pretext_lr']
    ENCODER_PARAMS = ['encoder_in_channels', 'encoder_hidden_channels', 'encoder_num_layers', 'encoder_dropout', 'encoder_heads']
    ALL_PARAMS = BENCHMARK_PARAMS + ENCODER_PARAMS

    GRAPH_PARAMS = ['nvertex', 'avg_degree', 'feature_center_distance', 'feature_dim',
       'edge_center_distance', 'edge_feature_dim', 'p_to_q_ratio',
       'num_clusters', 'cluster_size_slope', 'power_exponent', 'min_deg', 'marginal_param']

    frames = []
    for (ssl_model, encoder, scheme) in itertools.product(*[ssl_models, encoders, training_schemes]):
        column = f'{encoder}_{ssl_model}_{scheme}_{suffix}'
        param_cols = [(param, f'{encoder}_{ssl_model}_{scheme}_{param}') for param in ALL_PARAMS]
        pretext_col = f'{encoder}_{ssl_model}_{scheme}_pretext_'
        for col in df.columns:
            if not pretext_col in col:
                continue
            param_cols += [(col.replace(pretext_col, ''), col)]

        if not column in df.columns:
            continue
            
        df_model = df[[column]].rename(columns=lambda col: col.replace(column, suffix))
        for param, param_c in param_cols:
            df_model[param] = df[param_c] if param_c in df.columns else None 
        if include_graph_params:
            for param in GRAPH_PARAMS:
                df_model[param] = df[param]
        if include_tuning_metric:
            df_model['downstream_val_tuning_metrics'] = df[f'{encoder}_{ssl_model}_{scheme}_downstream_val_tuning_metrics']
        if include_embeddings:
            df_model['embeddings'] = df[f'{encoder}_{ssl_model}_{scheme}_embeddings']
            df_model['classes'] = df[f'{encoder}_{ssl_model}_{scheme}_classes']
        df_model['SSL_model'] = ssl_model
        df_model['SSL_category'] = ssl_method_to_category(ssl_model)
        df_model['Encoder'] = encoder
        df_model['Training_scheme'] = scheme
        df_model['Graph_ID'] = df.index.values.tolist()
        if 'Experiment' in df.columns:
            df_model['Experiment'] = df['Experiment']

        frames += [df_model]
    return pd.concat(frames, ignore_index=True)


def unpivot_baseline_model(df : pd.DataFrame, suffix : str, baseline_models, training_schemes, 
                           include_tuning_metric=False, include_graph_params=False, include_embeddings=False):
    '''
    Unpivot the results to a long format for all baseline methods. Each row corresponds to an experiment on graph.
    '''
    BENCHMARK_PARAMS = ['train_downstream_lr', 'train_pretext_weight',
               'train_pretext_epochs', 'train_pretext_lr']
    ENCODER_PARAMS = ['encoder_in_channels', 'encoder_hidden_channels', 'encoder_num_layers', 'encoder_dropout', 'encoder_heads']
    ALL_PARAMS = BENCHMARK_PARAMS + ENCODER_PARAMS

    GRAPH_PARAMS = ['nvertex', 'avg_degree', 'feature_center_distance', 'feature_dim',
       'edge_center_distance', 'edge_feature_dim', 'p_to_q_ratio',
       'num_clusters', 'cluster_size_slope', 'power_exponent', 'min_deg', 'marginal_param']

    frames = []
    for baseline_model, training_scheme in itertools.product(*[baseline_models, training_schemes]):
        column = f'{baseline_model}__{training_scheme}_{suffix}'
        param_cols = [(param, f'{baseline_model}__{training_scheme}_{param}') for param in ALL_PARAMS]
        if not column in df.columns:
            continue
        df_model = df[[column]].rename(columns=lambda col: col.replace(column, suffix))
        for param, param_c in param_cols:
            df_model[param] = df[param_c] if param_c in df.columns else None 
        if include_graph_params:
            for param in GRAPH_PARAMS:
                df_model[param] = df[param]
        if include_tuning_metric:
            df_model['downstream_val_tuning_metrics'] = df[f'{baseline_model}__{training_scheme}_downstream_val_tuning_metrics']
        if include_embeddings:
            df_model['embeddings'] = df[f'{baseline_model}__{training_scheme}_embeddings']
            df_model['classes'] = df[f'{baseline_model}__{training_scheme}_classes']
        df_model['Baseline_model'] = baseline_model
        df_model['Graph_ID'] = df.index.values.tolist()
        if 'Experiment' in df.columns:
            df_model['Experiment'] = df['Experiment']
        
        frames += [df_model]
    return pd.concat(frames, ignore_index=True)

def __deprecated_create_global_latex_table(df):
    for model, r in df.iterrows():
        lines = ['\\texttt{' + model + '}']
        lists = []
        for train in ['PF', 'URL', 'JL']:
            for enc in ['GCN', 'GAT', 'GIN']:
                line = r[(train, enc)]
                if pd.isna(line):
                    lines += ['-']
                    continue
                line = line.replace('±', '\pm')
                lines += [f' ${line}$']
                lists += [enc + '-' + train]
        print(' &'.join(lines))
        print('\\\\')

def create_encoder_latex_table(df_ssl_category_means, df_ssl_category_stds):
    for training_Scheme in ['PF', 'URL', 'JL']:
        for encoder in ['GCN', 'GAT', 'GIN']:
            query = (df_ssl_category_means.Training_scheme == training_Scheme) & (df_ssl_category_means.Encoder == encoder)
            mean = df_ssl_category_means.loc[query, TEST_METRIC].values[0]
            std = df_ssl_category_stds.loc[query, TEST_METRIC].values[0]
            print(f'${mean}\pm{std}$ ', end='')
        print()

