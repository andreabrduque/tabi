#do one by one

import concurrent.futures
import requests

from typing import Optional, List



#https://en.wikipedia.org/w/api.php?action=query&prop=pageprops|extracts&pageids=61421301&exsentences=10&exlimit=1&explaintext=1&formatversion=2

#if no extract and no pageprops, save thingie
#get claims with #https://www.wikidata.org/w/api.php?action=wbgetclaims&entity=Q180736&formatversion=2
#extract P31

LAST_KILT_E_WIKIPEDIA_ID = 61421301

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
    get_entities_endpoint = f"https://www.wikidata.org/w/api.php?action=wbgetentities&ids={wikidata_id}&props=labels&sitefilter=azwiki&formatversion=2&languages=en&format=json"
      
    r = requests.get(
            get_entities_endpoint
        )
    payload = r.json()
    # P31 is the attribute -instance of-
    instance_type_label = payload["entities"][wikidata_id]["labels"]["en"]["value"]
    
    return instance_type_label      
      
    
    
def get_entity_type_from_claims(wikidata_id: str) -> Optional[str]:
    claims_endpoint = f"https://www.wikidata.org/w/api.php?action=wbgetclaims&entity={wikidata_id}&formatversion=2&format=json"
    
    r = requests.get(
        claims_endpoint
    )
    payload = r.json()
    # P31 is the attribute -instance of-
    instance_type_id = payload["claims"]["P31"][0]["mainsnak"]["datavalue"]["value"]["id"]
    instance_type_label = entity_types_label(instance_type_id)
    
    return instance_type_label
       

    

#Get data from Wikidatas API
def query_and_parse_wikidata_api(page_id) -> Optional[dict]:
    query_endpoint = f"https://en.wikipedia.org/w/api.php?action=query&prop=pageprops|extracts&pageids={page_id}&exsentences=10&exlimit=1&explaintext=1&formatversion=2&format=json"
    
    try:
        r = requests.get(
            query_endpoint
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
    
    
if __name__ == "__main__":
    print(query_and_parse_wikidata_api(61421301))
    
