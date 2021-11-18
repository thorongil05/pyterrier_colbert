import unittest
import pandas as pd
import os
from pyterrier_colbert.info_pruning import InfoPruning
from pyterrier_colbert.pruning import blacklisted_tokens_transformer

class TestPruning(unittest.TestCase):

    def setUp(self):
        self.test_df = pd.read_pickle(os.path.join(os.getcwd(), "tests/test-resources/test.pkl"))

    def test_blacklist_transformer_token(self):
        '''
        Check if the token exists after pruning
        '''
        test_blacklist = [1997]
        transformer = blacklisted_tokens_transformer(self.test_df, test_blacklist)
        df = transformer.transform(self.test_df)
        self.assertFalse(self._term_occurs(1997, df))
        
    def test_blacklist_transformer_token_2(self):
        '''
        Check if the two tokens exist after pruning
        '''
        test_blacklist = [1997, 3280]
        transformer = blacklisted_tokens_transformer(self.test_df, test_blacklist)
        df = transformer.transform(self.test_df)
        self.assertFalse(self._term_occurs(1997, df) or self._term_occurs(3290, df))
    
    def test_blacklist_transformer_embs(self):
        '''
        Check if the embeddings related to the token exists after pruning
        '''
        test_blacklist = [1997]
        for i in range(len(self.test_df)):
            toks = self.test_df.iloc[i].doc_toks
            embs = self.test_df.iloc[i].doc_toks
            for i, tok in enumerate(toks):
                if tok == 1997: emb = embs[i]
        transformer = blacklisted_tokens_transformer(self.test_df, test_blacklist)
        df = transformer.transform(self.test_df)
        self.assertFalse(self._embs_occurs(emb, df))
        
    def test_blacklist_transformer_embs_2(self):
        '''
        Check if the two embeddings related to the tokens exist after pruning
        '''
        test_blacklist = [1997, 3280]
        for i in range(len(self.test_df)):
            toks = self.test_df.iloc[i].doc_toks
            embs = self.test_df.iloc[i].doc_toks
            for i, tok in enumerate(toks):
                if tok == 1997: emb_1 = embs[i]
                if tok == 3280: emb_2 = embs[i]
        transformer = blacklisted_tokens_transformer(self.test_df, test_blacklist)
        df = transformer.transform(self.test_df)
        self.assertFalse(self._embs_occurs(emb_1, df) or self._embs_occurs(emb_2, df))
        
    def test_blacklist_transformer_info_pruning(self):
        '''
        Check if the number of embeddings removed in info_pruning is correct.
        '''
        info_pruning = InfoPruning()
        test_blacklist = [1997]
        for i in range(len(self.test_df)):
            toks = self.test_df.iloc[i].doc_toks
            embs = self.test_df.iloc[i].doc_toks
            for i, tok in enumerate(toks):
                if tok == 1997: emb = embs[i]
        transformer = blacklisted_tokens_transformer(self.test_df, test_blacklist, info_pruning)
        df = transformer.transform(self.test_df)
        counter = self._count_token_ids(docid=11268, tid=1997)
        pruning_df = info_pruning.get_dataframe()
        n_embeddings_removed = pruning_df.loc[pruning_df['doc_id'] == 11268].embeddings_pruned.values
        self.assertTrue(n_embeddings_removed == counter)
        
    def _term_occurs(self, tid, df):
        '''
        Return true if the tid occurs in the df
        '''
        occurrences_tid = []
        for i in range(len(df)):
            n_occurrences = 0
            for element in df.iloc[i].doc_toks:
                if tid in element: n_occurrences += 1
            occurrences_tid.append(n_occurrences)
        if 1 in occurrences_tid: return True
        return False
    
    def _embs_occurs(self, emb, df):
        '''
        Return true if the embedding occurs in the df
        '''
        occurrences_emb = []
        for i in range(len(df)):
            n_occurrences = 0
            for element in df.iloc[i].doc_embs:
                if emb in element: n_occurrences += 1
            occurrences_emb.append(n_occurrences)
        if 1 in occurrences_emb: return True
        return False
    
    def _count_token_ids(self, docid, tid):
        '''
        Count the occurrences of a tid for a dataframe row identified by a docid
        '''
        row = self.test_df.loc[self.test_df['docid'] == docid]
        tokens = row.doc_toks.values[0]
        counter = 0
        for tok in tokens:
            if tok.item() == tid: counter += 1
        return counter