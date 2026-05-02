import pandas as pd
from datasets import load_dataset, get_dataset_config_names, DatasetDict


def dataset_to_dataframe(ds_obj, dataset_name: str, config_name=None) -> pd.DataFrame:
    dfs = []
    if isinstance(ds_obj, DatasetDict):
        for split_name, dset in ds_obj.items():
            tmp = dset.to_pandas()
            tmp["_split"] = split_name
            dfs.append(tmp)
    else:
        tmp = ds_obj.to_pandas()
        tmp["_split"] = "default"
        dfs.append(tmp)
    out = pd.concat(dfs, ignore_index=True)
    out["_dataset"] = dataset_name
    out["_config"] = config_name if config_name is not None else "default"
    return out


def load_all_configs(repo_id: str, dataset_name: str, token: str) -> pd.DataFrame:
    try:
        configs = get_dataset_config_names(repo_id, token=token)
    except Exception:
        configs = [None]
    if configs == ["default"]:
        configs = [None]
    dfs = []
    for cfg in configs:
        try:
            ds = load_dataset(repo_id, cfg, token=token) if cfg is not None else load_dataset(repo_id, token=token)
            dfs.append(dataset_to_dataframe(ds, dataset_name, cfg))
        except Exception as e:
            print(f"[WARN] failed to load {dataset_name}, config={cfg}: {repr(e)}")
    if not dfs:
        raise RuntimeError(f"Failed to load dataset: {repo_id}")
    return pd.concat(dfs, ignore_index=True)


def load_benchmarks(repo_ids: dict, token: str, max_rows_per_dataset=None, seed=42):
    dfs = []
    for name, repo_id in repo_ids.items():
        print(f"[LOAD] {name}: {repo_id}")
        df = load_all_configs(repo_id, name, token)
        if max_rows_per_dataset is not None:
            df = df.sample(n=min(max_rows_per_dataset, len(df)), random_state=seed).reset_index(drop=True)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)
