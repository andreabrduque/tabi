import argparse
import logging
from collections import defaultdict

from string import punctuation
import torch
from termcolor import colored
from transformers import AutoTokenizer
from transformers import logging as hf_logging

from tabi.constants import ENT_START, MENTION_END, MENTION_START
from tabi.models.biencoder import Biencoder
from tabi.utils.data_utils import load_entity_data
from tabi.utils.utils import load_model, move_dict
import tabi.utils.data_utils as data_utils
from typing import Dict, List

#API

import logging
from typing import Any
from fastapi import Depends, FastAPI
from typing import List, Tuple
from pydantic import BaseModel

class NerdInput(BaseModel):
    text: str
    entity_spans: List[int]
    k: int
    lang: str = "en"


max_context_length = 64
top_k = 5
entity_emb_path = "/Users/andrea/Documents/Workspace/tabi/data/embs.npy"
model_checkpoint = "/Users/andrea/Documents/Workspace/tabi/data/best_model.pth"
entity_file = "/Users/andrea/Documents/Workspace/tabi/data/entity.pkl"
device = "cpu"

hf_logging.set_verbosity_error()

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

def get_inputs(text, char_spans):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_special_tokens({"additional_special_tokens": [ENT_START, MENTION_START, MENTION_END]})


    context_tokens = data_utils.get_context_window(
        char_spans=char_spans,
        tokenizer=tokenizer,
        context=text,
        max_context_length=max_context_length,
    )

    # convert back to string to use tokenizer to pad and generate attention mask
    context = tokenizer.decode(
        tokenizer.convert_tokens_to_ids(context_tokens)
    )
    inputs = tokenizer(
        context,
        padding="max_length",
        add_special_tokens=True,
        return_tensors="pt",  # return as pytorch tensors
        truncation=True,
        max_length=max_context_length,
    )

    return inputs


# load model
logger.info("Loading model...")
model = Biencoder(
    tied=True,
    entity_emb_path=entity_emb_path,
    top_k=top_k,
    model_name="bert-base-uncased",
    normalize=True,
    temperature=0.05,
)
load_model(model_checkpoint=model_checkpoint, device=device, model=model)
model.to(device)
model.eval()
logger.info("Finished loading model!")

# create tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer.add_special_tokens({"additional_special_tokens": [ENT_START]})

# load entity cache
logger.info("Loading entity data...")
entity_cache = load_entity_data(entity_file)
logger.info("Finished loading entity data!")


def pretty_print(ent_data, prob, score):
    print(colored(f"\ntitle: {ent_data['title']}", "grey", "on_cyan"))
    print(f"prob: {round(prob, 5)}")
    print(f"score: {round(score, 5)}")
    print(f"text:{' '.join(ent_data['description'].split(' ')[:150])}")



LOGGING_FORMAT = "timestamp=%(asctime)s level=%(levelname)s name=%(name)s message=%(message)s"
logging.basicConfig(format=LOGGING_FORMAT, level=logging.INFO)
logger = logging.getLogger("nerdApp")

app = FastAPI()


@app.post(
    "/predict",
    response_model=Any,
    description="Route to perform named entity disambiguation"
)
def predict(payload: NerdInput) -> List[Dict]:
    inputs = get_inputs(payload.text, payload.entity_spans)

    with torch.no_grad():
        res = model.predict(
            context_data=move_dict(inputs, device),
            data_id=torch.tensor([-1]),
        )
        assert len(res["probs"]) == 1
        res["probs"] = res["probs"][0].tolist()
        res["indices"] = res["indices"][0].tolist()
        res["scores"] = res["scores"][0].tolist()
        del res["data_id"]
        
        result = []

        # return response to user
        for eid, prob, score in zip(res["indices"], res["probs"], res["scores"]):
            pretty_print(entity_cache[eid], prob, score)
            result.append(
                {
                "neighbor": entity_cache[eid],
                "prob": prob,
                "score": score
             }
            )

    return result

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_config=None)
