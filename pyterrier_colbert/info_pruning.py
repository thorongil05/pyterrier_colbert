import pandas as pd
import math
import json

class InfoPruning:
    '''
    Pruning Class
    '''
    
    def __init__(self):
        self.pruning_info = {}
        self.pruning_dataframes = []
        self.pruning_counter = 0
        self.n_docs = 0


    def add(self, query_id, doc_id, doc_len, embeddings_pruned):
        self.pruning_info[hash((query_id, doc_id))] = {
            'query_id': query_id,
            'doc_id': doc_id,
            'doc_len': doc_len, 
            'embeddings_pruned': embeddings_pruned
        }

    def get_dataframe(self):
        df = pd.DataFrame(self.pruning_info.values(), columns=['query_id', 'doc_id', 'doc_len', 'embeddings_pruned'])
        return df
    
    def get_blacklist(factory, path, verbose=False):
        # TODO: refactor the parameter factory (maybe I can pass directly faiss_nn_term)
        faiss_nn_term = factory.nn_term(df=True)
        vocabulary = faiss_nn_term.tok.get_vocab()
        n_docs = faiss_nn_term.num_docs
        if verbose:
            print(f'Number of docs: {n_docs}')
            print(f'Vocabulary Length: {len(vocabulary)}')
        with open(path) as f:
            stopwords = json.load(f)
        if verbose: print("Stopwords length:", len(stopwords))
        blacklist_tids = []

        for stopword in stopwords:
            if stopword in vocabulary:
                blacklist_tids.append(vocabulary[stopword])

        # Remove items with 0 document frequency
        if verbose: print("Blacklist length:", len(blacklist_tids))
        blacklist_tids_dfs = []
        for tid in blacklist_tids:
            df = factory.nn_term(df=True).getDF_by_id(tid)
            idf = math.log(n_docs/(df + 1), 10)
            if df != 0: blacklist_tids_dfs.append((tid, idf))
        if verbose: print("Blacklist length (without 0 df elements):", len(blacklist_tids_dfs))
        # order by inverse document frequency
        ordered_blacklist = sorted(blacklist_tids_dfs, key= lambda pair: pair[1])
        final_blacklist = []
        for _id, _ in ordered_blacklist: final_blacklist.append(_id)
        return final_blacklist
            
    def _get_pruning_info(self):
        rows = []
        for query_id, query_data in self.pruning_info.items():
            row = self._get_pruning_info_per_query_data(query_id, query_data)
            rows.append(row)
        df = pd.DataFrame(data=rows,
            columns=['qid', '# total embeddings', '# tokens pruned', 'tokens pruned %', 'most pruned document', 'less pruned document'])
        return df