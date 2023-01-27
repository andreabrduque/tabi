import numpy as np
import pickle
import jsonlines
import argparse


def merge_embeddings(entity_emb_path_1, entity_emb_path_2, dim):

    def read_embedding(path, dim):
        return np.copy(np.memmap(path, dtype="float32", mode="r").reshape(
            -1, dim
        ))
    entity_embs_1 = read_embedding(entity_emb_path_1, dim)
    entity_embs_2 = read_embedding(entity_emb_path_2, dim)
    return np.concatenate([entity_embs_1, entity_embs_2])


def save_embeddings(embeddings, path):
    rows_len, cols_len = embeddings.shape
    mmap_file = np.memmap(
        path, dtype="float32", mode="w+", shape=(rows_len, cols_len)
    )
    mmap_file[:] = embeddings[:]
    mmap_file.flush()
    print(f"Saved embeddings to {path} of shape {rows_len}x{cols_len}")


def merge_entity_caches(pickle_path, new_dataset, new_pickle_path):
    with open(pickle_path, "rb") as f:
        entity_map_pickle = pickle.load(f)
    ids_len = len(entity_map_pickle)
    print(f"Loaded pickle with {ids_len} items")
    with jsonlines.open(new_dataset) as f:
        for line_idx, line in enumerate(f):
            lid = ids_len + line_idx
            entity_map_pickle[lid] = {
                "id": lid,
                "description": line.get("text", ""),
                "title": line.get("title", ""),
                "types": line.get("types", []),
                "wikipedia_page_id": line.get("wikipedia_page_id", ""),
            }

    with open(new_pickle_path, "wb") as f:
        pickle.dump(entity_map_pickle, f)
        print(f"Wrote {len(entity_map_pickle)} ")


def main(args):
    embeds = merge_embeddings(args.embeddings_path_1,
                              args.embeddings_path_2, args.embeddings_dim)
    save_embeddings(embeds, args.embeddings_new_path)
    merge_entity_caches(args.entity_pickle_path,
                        args.entity_dataset_path, args.entity_pickle_new_path)


if __name__ == "__main__":
    # add arguments specific to entity extraction to parser
    parser = argparse.ArgumentParser(add_help=False)
    parser_args = parser.add_argument_group("parser_args")
    parser_args.add_argument("--entity_pickle_path", type=str, required=True)
    parser_args.add_argument(
        "--entity_pickle_new_path", type=str, required=True)
    parser_args.add_argument("--entity_dataset_path", type=str, required=True)
    parser_args.add_argument("--embeddings_dim", type=int, default=768)
    parser_args.add_argument("--embeddings_path_1", type=str, required=True)
    parser_args.add_argument("--embeddings_path_2", type=str, required=True)
    parser_args.add_argument("--embeddings_new_path", type=str, required=True)

    args = parser.parse_args()
    main(args)
