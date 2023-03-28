import json
import pandas as pd

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



def get_best_configuration_per_model(df, TEST_METRIC):
    best_configurations = {}

    # Common hparams
    COMMON_PARAMS = ['encoder_in_channels', 'encoder_hidden_channels', 'encoder_num_layers', 'encoder_dropout']

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
        best_configuration = means[test_metric].nlargest(1).reset_index()
        best_configurations[model] = best_configuration
    return best_configurations