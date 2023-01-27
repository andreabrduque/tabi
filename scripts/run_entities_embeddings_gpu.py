import subprocess

# set hyperparameters

add_types_in_desc = True
batch_size = 32
entity_batch_size = 32

# machine params
gpus = "0"
ngpus = 1
distributed = False

# set paths
home_dir = "tabi"
data_dir = "data"
entity_file = "test.jsonl"
type_file = "figer_types.txt"
embeddings = "embs_new.npy"

base_log_dir = "logs"
run_name = "sample_run"

log_dir = f"{base_log_dir}/{run_name}"

preprocess = True
tokenized_entity_data = f"{log_dir}/1_preprocess/entity_data.npy"

python_str = "python"
if distributed:
    python_str = f"python -m torch.distributed.launch --nproc_per_node={ngpus}"

# preprocess entity data (generate and save tokens for BERT entity input)
if preprocess:
    subprocess.run(
        f"python {home_dir}/preprocess_entity.py \
        --batch_size {batch_size} \
        --add_entity_type_in_description {add_types_in_desc} \
        --log_dir {log_dir}/1_preprocess \
        --type_file {data_dir}/{type_file} \
        --entity_file {data_dir}/{entity_file}",
        shell=True,
        check=True,
    )

# generate entities
subprocess.run(
    f"CUDA_VISIBLE_DEVICES={gpus} python {home_dir}/extract_entity.py \
    --entity_emb_path {embeddings} \
    --entity_file {data_dir}/{entity_file} \
    --model_checkpoint {data_dir}/best_model.pth \
    --batch_size {entity_batch_size} \
    --log_dir {log_dir}/1_entity \
    --add_entity_type_in_description {add_types_in_desc} \
    --distributed {distributed} \
    --tokenized_entity_data {tokenized_entity_data} \
    --type_file {data_dir}/{type_file}",
    shell=True,
    check=True,
)