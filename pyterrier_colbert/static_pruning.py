import math
import json
import torch
from pyterrier.transformer import TransformerBase
from pyterrier_colbert.faiss_term_index import FaissNNTerm
import os
import pyterrier as pt
import time
from pyterrier.measures import RR, nDCG, AP, MRR, R
from pyterrier_colbert.pruning import scorer, fetch_index_encodings
from pyterrier_colbert.ranking import ColBERTFactory
import ir_measures


def get_pruning_ratio(blacklist, faiss_nn_term : FaissNNTerm):
    # Returns the percentage of pruning: e.g. 50%
    ids_to_prune = torch.unique(torch.tensor(blacklist, dtype=torch.int32))
    embeddings_to_prune = torch.sum(torch.index_select(faiss_nn_term.lookup, 0, ids_to_prune))
    total_embeddings = torch.sum(faiss_nn_term.lookup)
    percentage_reduction_of_corpus = embeddings_to_prune / total_embeddings
    return round(percentage_reduction_of_corpus.item() * 100, 2)

def get_reduction(blacklist, faiss_nn_term : FaissNNTerm):
    # Returns the reduction value: e.g. 50% pruning means 2x, 75% pruning means 4x
    ids_to_prune = torch.unique(torch.tensor(blacklist, dtype=torch.int32))
    embeddings_to_prune = torch.sum(torch.index_select(faiss_nn_term.lookup, 0, ids_to_prune))
    total_embeddings = torch.sum(faiss_nn_term.lookup)
    percentage_reduction_of_corpus = embeddings_to_prune / total_embeddings
    return round(1/(1 - percentage_reduction_of_corpus.item()), 2)

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

def get_tid_ordered_by_idf(faiss_nn_term : FaissNNTerm, verbose=False):
        
        assert hasattr(faiss_nn_term, 'dfs'), 'FaissNNTerm class must be initialized with dfs=True'

        vocabulary = faiss_nn_term.tok.get_vocab()
        n_docs = faiss_nn_term.num_docs
        
        if verbose:
            print(f'Number of docs: {n_docs}')
            print(f'Vocabulary Length: {len(vocabulary)}')
       
        tokens = []
        for token in vocabulary:
            tid = vocabulary[token]
            df = faiss_nn_term.getDF_by_id(tid)
            if(df != 0):
                idf = math.log(n_docs/(df + 1), 10)
                tokens.append((tid, idf))
        
        # Remove items with 0 document frequency
        if verbose: print("Token length (without 0 df elements):", len(tokens))
        # order by inverse document frequency
        ordered_tokens = sorted(tokens, key= lambda pair: pair[1])
        final_token_list = [_id for _id, _ in ordered_tokens]
        return final_token_list

def get_stopwords_from_file(faiss_nn_term : FaissNNTerm, path):
    '''
    Get the tids from a json list of stopwords (path)
    '''
    assert hasattr(faiss_nn_term, 'dfs'), 'FaissNNTerm class must be initialized with dfs=True'
    vocabulary = faiss_nn_term.tok.get_vocab()
    with open(path) as f:
        stopwords = json.load(f)
    print(f'Loaded {len(stopwords)} stopwords')
    tids = []
    for term in stopwords:
        if term in vocabulary:
            tid = vocabulary[term]
            tids.append(tid)
    print(f'tids found for those stopwords: {len(tids)}')
    return tids

def run_single_experiment(pipeline, batch_size, name, topics, qrels, save_dir):
    start_time = time.time()
    df_experiment = pt.Experiment(
        [pipeline],
        topics,
        qrels,
        batch_size=batch_size,
        filter_by_qrels=True,
        eval_metrics=[RR(rel=2), nDCG@10, nDCG@100, AP(rel=2), 'recip_rank', RR@10, MRR@10],
        names=[name],
        save_dir=save_dir,
        verbose=True
    )
    time_elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))    
    return df_experiment, time_elapsed

def get_pipeline(factory, blacklist, k1):
    pipeline = (
        factory.query_encoder()
        >> (factory.ann_retrieve_score(query_encoded=True) % k1)
        >> fetch_index_encodings(factory, ids=True, verbose=False)
        >> blacklisted_tokens_transformer(blacklist, verbose=False)
        >> scorer(factory, verbose=False)
        >> pt.apply.doc_embs(drop=True)
        >> pt.apply.query_embs(drop=True)
    )
    return pipeline

def blacklist_experiment(factory: ColBERTFactory, k1, blacklist, name, short_name, topics, qrels, save_dir, batch_size=50):
    faiss_nn_term = factory.nn_term(df=True)
    torch.cuda.empty_cache()
    pipeline = get_pipeline(factory, blacklist, k1)
    df_experiment, time_elapsed = run_single_experiment(pipeline, batch_size, name, topics, qrels, save_dir)
    df_experiment = df_experiment.set_index('name')
    idx_pruning = get_pruning_ratio(blacklist, faiss_nn_term)
    df_experiment['% index pruning'] = idx_pruning
    if idx_pruning != 0:
        df_experiment['reduction'] = get_reduction(blacklist, faiss_nn_term)
    else:
        df_experiment['reduction'] = 1
    df_experiment['short-name'] = short_name
    send_notification(f'Index Pruning: {idx_pruning} for {short_name} in {time_elapsed}')
    return df_experiment

def send_notification(message):
    url = 'https://pytraining-bot.herokuapp.com/notify'
    chat_id = 182166729
    body = {'message': message, 'chat_id': chat_id}
    import requests
    requests.post(url = url, data=body)

# New measures

def _avg_doc_len(qrels, run):
    from warnings import warn
    if 'doc_toks' not in run.columns:
        if len ( run )  > 0:
            warn("run of %d rows did not have doc_toks column; available columns: %s" % (len ( run ), str(run.columns)))
        else:
            warn("empty run did not have doc_toks column; available columns: %s" % str(run.columns))
        return 0
    return run['doc_toks'].apply(lambda row: row.shape[0]).mean()
    

AvgDocLen = ir_measures.define_byquery(
    _avg_doc_len, 
    name="AvgDoclen")

MEASURES = [AP(rel=2)@1000, nDCG@10,nDCG@20,nDCG@100, RR(rel=2)@10,RR(rel=2),R(rel=2)@1000,"mrt", AvgDocLen@1, AvgDocLen@10, AvgDocLen@100, AvgDocLen]