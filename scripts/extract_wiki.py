#do one by one

import requests

from typing import Optional, List
import pandas as pd
import argparse
import logging

WIKIPEDIA_URL="https://en.wikipedia.org/w/api.php"
WIKIDATA_URL="https://www.wikidata.org/w/api.php"

logger = logging.getLogger()


def new_entity_line(title: str, text: str, types_attempt: str, entity_types: List[str], page_id: int, kb_idx: str):
    return {
        "title": title, # title of the entity 
        "text": text, # description of the entity 
        "types_attempt": types_attempt, # my own attempt of getting types
        "types": entity_types, # list of types ([] if none) 
        "wikipedia_page_id": page_id, # wikipedia page id (can exclude if not linking to Wikipedia),
        "kb_idx":  kb_idx, #wikidata QID,
        "idx": f"https://en.wikipedia.org/wiki?curid={page_id}" #Link to wikipedia page        
    }
    
    
def entity_types_label(wikidata_id: str) -> str:     
    r = requests.get(WIKIDATA_URL,
                 params = {
                     "action": "wbgetentities",
                     "format": "json",
                     "ids": wikidata_id,
                     "props": "labels",
                     "sitefilter": "azwiki",
                     "formatversion": 2,
                     "languages": "en"
                 }
            )  
      
    payload = r.json()
    # P31 is the attribute -instance of-
    instance_type_label = payload["entities"][wikidata_id]["labels"]["en"]["value"]
    
    return instance_type_label      
      
    
    
def get_entity_type_from_claims(wikidata_id: str) -> Optional[str]:   
    r = requests.get(WIKIDATA_URL,
                 params = {
                     "action": "wbgetclaims",
                     "format": "json",
                     "entity": wikidata_id,
                     "sitefilter": "azwiki",
                     "formatversion": 2,
                 }
            )  
    
    
    payload = r.json()
    # P31 is the attribute -instance of-
    instance_type_id = payload["claims"]["P31"][0]["mainsnak"]["datavalue"]["value"]["id"]
    instance_type_label = entity_types_label(instance_type_id)
    
    return instance_type_label
       

    

#Get data from Wikidatas API
def query_and_parse_wikidata_api(page_id) -> Optional[dict]:   
    try:
        r = requests.get(WIKIPEDIA_URL,
                 params = {
                     "action": "query",
                     "format": "json",
                     "prop": "pageprops|extracts",
                     "exsentences": 10,
                     "exlimit": 1,
                     "explaintext": 1,
                     "formatversion": 2,
                     "pageids": page_id,
                     "sitefilter": "azwiki",
                     "formatversion": 2,
                 }
        )  
        payload = r.json()
        pages = payload["query"]["pages"]
        
        if(len(pages) > 1):
            return None
        
        item = pages[0]
        page_id = item["pageid"]
        
        if(not ("pageprops" in item.keys() and "extract" in item.keys())):
            return None
        
        wikidata_id =  item["pageprops"]["wikibase_item"]        
        text = item["extract"]
        title = item["title"]
        type_attempt = get_entity_type_from_claims(wikidata_id)
              
        return new_entity_line(title, text, type_attempt, [], page_id, wikidata_id)         
        
        
    except Exception as e:
        print(e)    
    
 
def main(args):
    data = []
    start = args.id_start
    entities_count = 0
    while(entities_count < args.entities_count):
        id = start + entities_count
        logger.info(f"Querying wikidata page {id}")
        maybe_entity = query_and_parse_wikidata_api(id)
        if(maybe_entity is not None):
            entity_title = maybe_entity["title"]
            logger.info(f"Found entity {entity_title}")
            data.append(maybe_entity)
            entities_count += 1
            
    df = pd.DataFrame(data)
    df.to_json(path_or_buf=args.output_file, orient='records', lines=True)
        
        

if __name__ == "__main__":
    # add arguments specific to entity extraction to parser
    parser = argparse.ArgumentParser(add_help=False)
    parser_args = parser.add_argument_group("parser_args")
    parser_args.add_argument("--output_file", type=str, required=True)
    parser_args.add_argument("--id_start", type=int, default=61421302)
    parser_args.add_argument("--entities_count", type=int, default=10)

    args = parser.parse_args()
    main(args)
