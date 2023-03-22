import json
import pandas as pd

def read_processed_shards(PROCESSED_DIR):
    with open(f'{PROCESSED_DIR}/summary.json', 'r') as f:
        summary = json.load(f)

    dfs = []
    for shard_idx in range(summary['N_RUNS']):
        filename = f'{shard_idx+1}.ndjson'

        with open(f'{PROCESSED_DIR}/shards/{filename}', 'r') as f:
            lines = f.readlines()
            records = map(json.loads, lines)
            dfs.append(pd.DataFrame.from_records(records))

    # Construct df
    results_df = pd.concat(dfs)
    del dfs
    return results_df