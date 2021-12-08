import pandas as pd
import math
import numpy as np
import json
import torch
from pyterrier.transformer import TransformerBase
from pyterrier_colbert.faiss_term_index import FaissNNTerm

def get_pruning_ratio(blacklist, faiss_nn_term : FaissNNTerm):
    ids_to_prune = torch.unique(torch.tensor(blacklist, dtype=torch.int32))
    embeddings_to_prune = torch.sum(torch.index_select(faiss_nn_term.lookup, 0, ids_to_prune))
    total_embeddings = torch.sum(faiss_nn_term.lookup)
    percentage_reduction_of_corpus = embeddings_to_prune / total_embeddings
    return round(percentage_reduction_of_corpus.item() * 100, 2)

def blacklisted_tokens_transformer(blacklist, verbose=False) -> TransformerBase:
    """
    Remove tokens and their embeddings from the document dataframe
    input: qid, query_embs, docno, doc_embs, doc_toks
    output: qid, query_embs, docno, doc_embs, doc_toks
    
    The blacklist parameters must contain a list of tokenids that should be removed
    """
    import pyterrier as pt
    import torch
    import numpy as np
    
    assert pt.started(), 'PyTerrier must be started'
    
    if torch.cuda.is_available(): 
        blacklist = torch.Tensor(blacklist).cuda()
    else:
        blacklist = torch.Tensor(blacklist)
        
    pt.tqdm.pandas()
    
    if verbose: print(f'Blacklist composed of {len(blacklist)} elements.')
        
    def _prune_gpu(row):
        tokens = row['doc_toks'].cuda()
        embeddings = row['doc_embs'].cuda()
        row_embs_size = embeddings.size()
        tokens_size = tokens.size()[0]
        
        # create the 1-D mask
        final_mask = torch.zeros(row_embs_size[0], dtype=torch.bool)
        final_mask[0:tokens_size] = torch.any(tokens[None, :] == blacklist[:, None], axis=0)
        
        # apply the mask
        row['doc_toks'][final_mask[0:tokens_size]] = 0
        row['doc_embs'][final_mask, :] = 0 
        return row
        
    def _prune(row):
        tokens = row['doc_toks']
        embeddings = row['doc_embs']
        row_embs_size = embeddings.size()
        tokens_size = tokens.size()[0]
                                        
        # create the 1-D mask
        final_mask = torch.zeros(row_embs_size[0], dtype=torch.bool)
        final_mask[0:tokens_size] = torch.any(tokens[None, :] == blacklist[:, None], axis=0)
            
        # apply the mask
        row['doc_toks'][final_mask[0:tokens_size]] = 0
        row['doc_embs'][final_mask, :] = 0
        return row
    
    prune_function = _prune_gpu if torch.cuda.is_available() else _prune

    def _apply(df):
        if verbose:
            df = df.progress_apply(prune_function, axis=1)
        else:
            df = df.apply(prune_function, axis=1)
        return df

    return pt.apply.generic(_apply)