import subprocess

# set hyperparameters

# number of epochs for each round of sampling
n_epochs = 1

# first epoch is in-batch negatives
num_neg_rounds = 3

# tabi-specific
type_weight = 0.1

# model params
max_context_length = 32
lr = 1e-5
temperature = 0.05
add_types_in_desc = False
seed = 1234
batch_size = 32
eval_batch_size = 32
neg_sample_batch_size = 32
entity_batch_size = 32
num_negatives = 1  # number of hard negatives to use for training
num_negatives_orig = 20  # total number of hard negatives to fetch (fetch extra since we filter gold ids and optionally based on counts)
filter_negatives = False  # whether to filter hard negatives

# machine params
distributed = False

# set paths
home_dir = "tabi"
data_dir = "data"
train_file = "train.jsonl"
dev_file = "dev.jsonl"
test_file = "dev.jsonl"
entity_file = "test.jsonl"
type_file = "figer_types.txt"
embeddings = "embs_test.npy"

base_log_dir = "logs"
run_name = "sample_run"

log_dir = f"{base_log_dir}/{run_name}"

preprocess = True
tokenized_entity_data = f"{log_dir}/1_preprocess/entity_data.npy"

python_str = "python"


# preprocess entity data (generate and save tokens for BERT entity input)
if preprocess:
    subprocess.run(
        f"python {home_dir}/preprocess_entity.py \
        --add_entity_type_in_description {add_types_in_desc} \
        --log_dir {log_dir}/1_preprocess \
        --type_file {data_dir}/{type_file} \
        --entity_file {data_dir}/{entity_file}",
        shell=True,
        check=True,
    )

# generate entities
subprocess.run(
    f"python {home_dir}/extract_entity.py \
    --entity_emb_path {embeddings} \
    --entity_file {data_dir}/{entity_file} \
    --model_checkpoint {data_dir}/best_model.pth \
    --log_dir {log_dir}/1_entity \
    --add_entity_type_in_description {add_types_in_desc} \
    --distributed {distributed} \
    --tokenized_entity_data {tokenized_entity_data} \
    --type_file {data_dir}/{type_file}",
    shell=True,
    check=True,
)