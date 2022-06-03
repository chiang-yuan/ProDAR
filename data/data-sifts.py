#!/usr/bin/env python
# coding: utf-8

# # **Data-SIFTS**
# This is the data-preprocessing script that screens through [RCSB PDB](https://www.rcsb.org/) [search API](https://search.rcsb.org/) and [PDBe](https://www.ebi.ac.uk/pdbe/) [SIFTS API](https://www.ebi.ac.uk/pdbe/api/doc/sifts.html) to retrieve targeted PDB ids and associated gene ontology (GO) codes.
# 
# ## **0. Prerequisites**
# ### Required packages

# In[1]:


import math
import time
from datetime import datetime
import numpy as np
import requests
import json
from tqdm.auto import tqdm

from argparse import Namespace


# ### Configurations

# In[2]:


config = Namespace(
    sifts_timeout=10,
    sifts_savedir='./sifts',
    go_thres=50
)


# ## **1. First screening**
# 
# ### Filter PDB ids and chains from [RCSB PDB](https://www.rcsb.org/) [search API](https://search.rcsb.org/)

# In[3]:


request = """{
  "query": {
    "type": "group",
    "logical_operator": "and",
    "nodes": [
      {
        "type": "terminal",
        "service": "text",
        "parameters": {
          "operator": "exact_match",
          "negation": false,
          "value": "Protein",
          "attribute": "entity_poly.rcsb_entity_polymer_type"
        }
      },
      {
        "type": "terminal",
        "service": "text",
        "parameters": {
          "operator": "range",
          "negation": false,
          "attribute": "rcsb_polymer_entity.formula_weight",
          "value": {
            "from": 10,
            "include_lower": true,
            "to": 300,
            "include_upper": true
          }
        }
      }
    ]
  },
  "request_options": {
    "results_verbosity": "minimal",
    "return_all_hits": true,
    "group_by": {
      "aggregation_method": "sequence_identity",
      "similarity_cutoff": 95,
      "ranking_criteria_type": {
        "sort_by": "entity_poly.rcsb_sample_sequence_length",
        "direction": "desc"
      }
    },
    "group_by_return_type": "representatives",
    "scoring_strategy": "combined",
    "sort": [
      {
        "sort_by": "rcsb_entry_info.resolution_combined",
        "direction": "asc"
      }
    ]
  },
  "return_type": "polymer_entity"
}"""

url = "https://search.rcsb.org/rcsbsearch/v1/query?json={:s}".format(request)
print("Fetching data from {:s}...".format(url))
data = requests.get(url)

if data.status_code != 200:
    raise AssertionError("cannot fetch data from databse")

decoded = data.json()

now = datetime.now()
dt_string = now.strftime("%m/%d/%Y %H:%M:%S")

print("\t{:d} out of total {:d} proteins received on {:s}".format(
    decoded['group_by_count'], 
    decoded['total_count'], 
    dt_string))

polymers = []
for entry in decoded['result_set']:
    polymers.append(entry['identifier'])

del data, decoded


# ## **2. Second screening**
# ### Fetch GO codes from [PDBe](https://www.ebi.ac.uk/pdbe/) [SIFTS API](https://www.ebi.ac.uk/pdbe/api/doc/sifts.html)

# In[4]:


print("Fetching GO codes...")

# t_str = time.time()

with open(f'{config.sifts_savedir}/sifts-err-1.log', "w") as err:
    mf_go_codes_all = []
    pdb_chains_all = []
    for i, polymer in enumerate(tqdm(polymers)):
        pdb_id, entity_id = polymer.lower().split('_')
    
        url = "https://www.ebi.ac.uk/pdbe/api/mappings/go/"+pdb_id

        try:
            data = requests.get(url, timeout=config.sifts_timeout)
        except requests.Timeout:
            err.write("\tpdb_id: {:4s} -> Timeout\n".format(pdb_id))
            err.flush()
            continue

        if data.status_code != 200:
            err.write("\tpdb_id: {:4s} -> Failure (status = {:d})\n".format(pdb_id, data.status_code))
            err.flush()
            continue

        decoded = data.json()

        for go_code in decoded[pdb_id]['GO'].keys():
            if decoded[pdb_id]['GO'][go_code]['category'] == "Molecular_function":
                for mapping in decoded[pdb_id]['GO'][go_code]['mappings']:
                    if int(mapping['entity_id']) == int(entity_id):
                        mf_go_codes_all.append(go_code)
                        pdb_chains_all.append('{:s}_{:s}'.format(pdb_id, mapping['chain_id']))

    err.close()


# ### Save all and filtered pdb-chain ids and GO codes

# In[5]:


pdb_chains = np.unique(pdb_chains_all)

np.savetxt(f'{config.sifts_savedir}/pdb_chains.dat', pdb_chains, fmt='%s')

uniques, counts = np.unique(mf_go_codes_all, return_counts=True)

np.savetxt(f'{config.sifts_savedir}/mf_go_codes-allcnt.dat', 
           np.concatenate((uniques.reshape(-1,1), 
                           counts.reshape(-1,1)),
                           axis=1),
           fmt='%s\t%s')

mask = counts >= config.go_thres

np.savetxt(f'{config.sifts_savedir}/mf_go_codes-thres-{config.go_thres:d}.dat', 
           np.concatenate((uniques[mask].reshape(-1,1), 
                           counts[mask].reshape(-1,1)),
                           axis=1),
           fmt='%s\t%s')

np.save(f'{config.sifts_savedir}/mf_go_codes-thres-{config.go_thres:d}.npy', 
        uniques[counts >= config.go_thres])


# ## **3. Export PDB-GO pairs**
# ### Generate PDB-GO pairs based on filtered entries

# In[7]:


print("Generating pdb mf-go vectors...")

mf_go_codes = np.load(f'{config.sifts_savedir}/mf_go_codes-thres-{config.go_thres:d}.npy')

pdbmfgos = {}

with open(f'{config.sifts_savedir}/sifts-err-2.log', "w") as err:
    for i, pdb_chain in enumerate(tqdm(pdb_chains)):
        pdb_id, chain_id = pdb_chain.split('_')

        url = "https://www.ebi.ac.uk/pdbe/api/mappings/go/"+pdb_id

        try:
            data = requests.get(url, timeout=config.sifts_timeout)
        except requests.Timeout:
            err.write("\tpdb_id: {:4s} -> Timeout\n".format(pdb_chain))
            err.flush()
            continue

        if data.status_code != 200:
            err.write("\tpdb_id: {:4s} -> Failure (status = {:d})\n".format(pdb_id, data.status_code))
            err.flush()
            continue

        decoded = data.json()

        pdbmfgo = np.zeros(mf_go_codes.shape, dtype=int)
        for go_code in decoded[pdb_id]['GO'].keys():
            if decoded[pdb_id]['GO'][go_code]['category'] == "Molecular_function":
                for mapping in decoded[pdb_id]['GO'][go_code]['mappings']:
                    if mapping['chain_id'] == chain_id:
                        pdbmfgo += np.where(mf_go_codes == go_code, 1, 0).astype(int)
    
        pdbmfgos[pdb_chain] = np.argwhere(np.where(pdbmfgo > 0, 1, 0)).reshape(-1).tolist()

    err.close()
    


# ### Save as JSON file

# In[8]:


with open(f'{config.sifts_savedir}/pdbmfgos-thres-{config.go_thres:d}.json', "w") as out_file:
    json.dump(pdbmfgos, out_file, 
              skipkeys=False, 
              ensure_ascii=True, 
              indent=None, separators=(', ', ': '),
              sort_keys=True)


# ### Dataset summary

# In[9]:


with open(f'{config.sifts_savedir}/pdbmfgos-thres-{config.go_thres:d}.json', "r") as in_file:
    pdbmfgos = json.load(in_file)

print("protein entries: {:6d}\nmf-go entries: {:6d}".format(
    len(pdbmfgos.keys()), 
    max(list(map(lambda x: [max(x)] if x else [], list(pdbmfgos.values()))))[0]+1)
)

