from pyterrier_colbert.ranking import ColBERTFactory
from pyterrier.transformer import TransformerBase
from pyterrier.measures import AP, nDCG, RR, R
from warnings import warn
from typing import Tuple, Callable
import pyterrier as pt
import pandas as pd
import ir_measures
import torch
import time
import json
import math

class StaticPruningFramework:

    def __init__(self):
        self.factory : ColBERTFactory = None
        self.topics_name : str  = ''
        self.topics : pd.DataFrame = None
        self.qrels : pd.DataFrame = None
        self.blacklist : torch.Tensor = None
        self.batch_size : int = 50
        self.measures = self._initialize_measures()

    def setup(self, dataset_name: str, topics_type: str, index: Tuple, checkpoint: str, save_dir: str = None):
        print('Initializing the ColBERT environment...')
        print(index)
        print(*index)
        print(len(index))
        if len(index) != 2:
            raise ValueError('The index variable must be a tuple composed by the index folder path and the index file name')
        if not pt.started(): pt.init()
        dataset = pt.get_dataset(dataset_name)
        self.topics = dataset.get_topics(topics_type)
        self.qrels =  dataset.get_qrels(topics_type)
        self.factory = ColBERTFactory(checkpoint, *index)
        self.faiss_nn_term = self.factory.nn_term(df=True)
        self.k1 = 1000 # the number of documents retrieved in ann phase
        self.save_dir = save_dir
        torch.cuda.empty_cache()
        print('Cuda cache cleaned.')
        print('ColBERT environment initialized')

    def initialize_blacklist(self, blacklist : torch.Tensor):
        self.blacklist = blacklist
        self.index_pruning_percentage, self.index_reduction = self._compute_index_pruning_values()

    def initialize_pipeline(self, pipeline : TransformerBase = None):
        if pipeline is not None: self.pipeline = pipeline
        pipeline = (
            self.factory.query_encoder()
            >> (self.factory.ann_retrieve_score(query_encoded=True) % self.k1)
            >> self.fetch_index_encodings(ids=True, verbose=False)
            >> self.blacklisted_tokens_transformer(self.blacklist, verbose=False)
            >> self.scorer(verbose=False)
            >> pt.apply.doc_embs(drop=True)
            >> pt.apply.query_embs(drop=True)
        )
        self.pipeline = pipeline

    def single_run(self, name, short_name, verbose=False, notification_function: Callable = None):
        # notification_function is a function to send a notification when the experiment ends.
        start_time = time.time()
        df_experiment = pt.Experiment(
            [self.pipeline],
            self.topics,
            self.qrels,
            batch_size=self.batch_size,
            filter_by_qrels=True,
            eval_metrics=self.measures,
            names=[name],
            save_dir=self.save_dir,
            verbose=verbose
        )
        time_elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        df_experiment = df_experiment.set_index('name')
        df_experiment['% index pruning'] = self.index_pruning_percentage
        df_experiment['reduction'] = self.index_reduction
        df_experiment['short-name'] = short_name
        if not notification_function is None:
            message = f'Experiment {name} completed in {time_elapsed} with a reduction of {self.index_reduction}x'
            notification_function(message)
        return df_experiment

    def run_baseline(self, notification_function : Callable, verbose=False):
        colbert_ann = ((self.factory.ann_retrieve_score(query_encoded=False, verbose=False) % self.k1)
               >> self.factory.index_scorer(query_encoded=True))
        name, short_name = 'colbert-base', 'base'
        start_time = time.time()
        df_base = pt.Experiment(
            [colbert_ann],
            self.topics,
            self.qrels,
            batch_size=self.batch_size,
            filter_by_qrels=True,
            eval_metrics=self.measures,
            names=[name],
            save_dir=self.save_dir,
            verbose=verbose
        )
        time_elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        df_base = df_base.set_index('name')
        df_base['% index pruning'] = self.index_pruning_percentage
        df_base['reduction'] = self.index_reduction
        df_base['short-name'] = short_name
        if not notification_function is None:
            message = f'Experiment {name} completed in {time_elapsed} with a reduction of {self.index_reduction}x'
            notification_function(message)
        return df_base

    def get_stopwords_from_file(self, path):
        '''
        Get the tids from a json list of stopwords (path)
        '''
        assert hasattr(self.faiss_nn_term, 'dfs'), 'FaissNNTerm class must be initialized with dfs=True'
        vocabulary = self.faiss_nn_term.tok.get_vocab()
        with open(path) as f:
            stopwords = json.load(f)
        print(f'Loaded {len(stopwords)} stopwords')
        tids = []
        for term in stopwords:
            if term in vocabulary:
                tid = vocabulary[term]
                tids.append(tid)
        print(f'tids found for those stopwords: {len(tids)}')
        return torch.tensor(tids)

    def get_tid_ordered_by_idf(self, verbose=False):
        
        assert hasattr(self.faiss_nn_term, 'dfs'), 'FaissNNTerm class must be initialized with dfs=True'

        vocabulary = self.faiss_nn_term.tok.get_vocab()
        n_docs = self.faiss_nn_term.num_docs
        
        if verbose:
            print(f'Number of docs: {n_docs}')
            print(f'Vocabulary Length: {len(vocabulary)}')
       
        tokens = []
        for token in vocabulary:
            tid = vocabulary[token]
            df = self.faiss_nn_term.getDF_by_id(tid)
            if(df != 0):
                idf = math.log(n_docs/(df + 1), 10)
                tokens.append((tid, idf))
        
        # Remove items with 0 document frequency
        if verbose: print("Token length (without 0 df elements):", len(tokens))
        # order by inverse document frequency
        ordered_tokens = sorted(tokens, key= lambda pair: pair[1])
        final_token_list = [_id for _id, _ in ordered_tokens]
        return final_token_list   
    
    ## Private Methods

    def _initialize_measures(self):
        def _avg_doc_len(qrels, run):
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
        MEASURES = [
            AP(rel=2)@1000, 
            nDCG@10,
            nDCG@20,
            nDCG@100, 
            RR(rel=2)@10,
            RR(rel=2),
            R(rel=2)@1000, 
            RR@10, 
            "mrt", 
            AvgDocLen@1, 
            AvgDocLen@10, 
            AvgDocLen@100, 
            AvgDocLen
        ]
        self.measures = MEASURES

    def _compute_index_pruning_values(self):
        # Returns a tuple containing:
        # - the percentage of pruning: e.g. 50%
        # - the index reduciton: e.g. 2x
        ids_to_prune = torch.unique(torch.tensor(self.blacklist, dtype=torch.int32))
        embeddings_to_prune = torch.sum(torch.index_select(self.faiss_nn_term.lookup, 0, ids_to_prune))
        total_embeddings = torch.sum(self.faiss_nn_term.lookup)
        ratio = (embeddings_to_prune / total_embeddings).item()
        percentage = round(ratio * 100, 2)
        if ratio != 1:
            # Here we are sure to not divide by 0
            reduction = round(1/(1 - ratio), 2)
        else:
            reduction = 1
        return percentage, reduction

    ## Transformers

    def blacklisted_tokens_transformer(self, blacklist, verbose=False) -> TransformerBase:
        """
        Remove tokens and their embeddings from the document dataframe
        input: qid, query_embs, docno, doc_embs, doc_toks
        output: qid, query_embs, docno, doc_embs, doc_toks
        
        The blacklist parameters must contain a list of tokenids that should be removed
        """
        
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

        def _apply(df: pd.DataFrame):
            print('Columns before: ' + df.columns)
            if verbose:
                df = df.progress_apply(prune_function, axis=1)
            else:
                df = df.apply(prune_function, axis=1)
            print('Columns after: ' + df.columns)
            return df

        return pt.apply.generic(_apply)

    def scorer(self, verbose=False) -> TransformerBase:
        """
        Calculates the ColBERT max_sim operator using previous encodings of queries and documents
        input: qid, query_embs, [query_weights], docno, doc_embs
        output: ditto + score, [+ contributions]
        """
        import torch
        import pyterrier as pt
        assert pt.started(), 'PyTerrier must be started'
        pt.tqdm.pandas()
        cuda0 = torch.device('cuda:0')

        def _build_interaction(row, D):
            doc_embs = row.doc_embs
            doc_len = doc_embs.shape[0]
            D[row.row_index, 0:doc_len, :] = doc_embs
        
        def _score_query(df):
            weightsQ = None
            Q = torch.cat([df.iloc[0].query_embs]).cuda()
            if "query_weights" in df.columns:
                weightsQ = df.iloc[0].query_weights.cuda()
            else:
                weightsQ = torch.ones(Q.shape[0]).cuda()        
            D = torch.zeros(len(df), self.factory.args.doc_maxlen, self.factory.args.dim, device=cuda0)
            df['row_index'] = range(len(df))
            if verbose:
                df.progress_apply(lambda row: _build_interaction(row, D), axis=1)
            else:
                df.apply(lambda row: _build_interaction(row, D), axis=1)
            maxscoreQ = (Q @ D.permute(0, 2, 1)).max(2).values
            scores = (weightsQ*maxscoreQ).sum(1).cpu()
            df["score"] = scores.tolist()
            df = self.factory._add_docnos(df)
            return df
            
        return pt.apply.by_query(_score_query)

    def fetch_index_encodings(self, verbose=False, ids=False) -> TransformerBase:
        """
        New encoder that gets embeddings from rrm and stores into doc_embs column.
        If ids is True, then an additional doc_toks column is also added. This requires 
        a Faiss NN term index data structure, i.e. indexing should have ids=True set.
        input: docid, ...
        output: ditto + doc_embs [+ doc_toks]
        """
        def _get_embs(df):
            rrm = self.factory._rrm() # _rrm() instead of rrm because we need to check it has already been loaded.
            if verbose:
                import pyterrier as pt
                pt.tqdm.pandas()
                df["doc_embs"] = df.docid.progress_apply(rrm.get_embedding) 
            else:
                df["doc_embs"] = df.docid.apply(rrm.get_embedding)
            return df

        def _get_tok_ids(df):
            fnt = self.factory.nn_term(False)
            def _get_toks(pid):
                end = fnt.end_offsets[pid]
                start = end - fnt.doclens[pid]
                return fnt.emb2tid[start:end].clone()

            if verbose:
                import pyterrier as pt
                pt.tqdm.pandas()
                df["doc_toks"] = df.docid.progress_apply(_get_toks)
            else:
                df["doc_toks"] = df.docid.apply(_get_toks)
            return df
        rtr = pt.apply.by_query(_get_embs, add_ranks=False)
        if ids:
            rtr = rtr >> pt.apply.by_query(_get_tok_ids, add_ranks=False)
        return rtr