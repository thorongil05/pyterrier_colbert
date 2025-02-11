import unittest
import pandas as pd
import os
from os.path import join
import torch
import json
from pyterrier_colbert.static_pruning import blacklisted_tokens_transformer

class TestPruning(unittest.TestCase):

    def setUp(self):
        self.test_df = pd.read_pickle(f"{os.getcwd()}/tests/test-resources/test.pkl")
        with open(join(os.getcwd(), 'tests', 'test-resources', 'blacklist.json')) as f:
            self.blacklist = json.load(f)

    def test_blacklist_transformer_token(self):
        '''
        Check if the token exists after pruning
        '''
        test_blacklist = [1997]
        transformer = blacklisted_tokens_transformer(test_blacklist)
        df = transformer.transform(self.test_df)
        self.assertFalse(self._term_occurs(1997, df))
        
    def test_blacklist_transformer_token_2(self):
        '''
        Check if the two tokens exist after pruning
        '''
        test_blacklist = [1997, 3280]
        transformer = blacklisted_tokens_transformer(test_blacklist)
        df = transformer.transform(self.test_df)
        self.assertFalse(self._term_occurs(1997, df) or self._term_occurs(3280, df))

    def test_blacklist_transformer_token_3(self):
        '''
        Check if the token list exist after pruning
        '''
        transformer = blacklisted_tokens_transformer(self.blacklist)
        df = transformer.transform(self.test_df)
        for token in self.blacklist:
            if self._term_occurs(token, df):
                self.assertTrue(False)
        self.assertTrue(True)

    def test_blacklist_transformer_token_4(self):
        '''
        Check if the token not present in the blacklist is not deleted after pruning
        '''
        transformer = blacklisted_tokens_transformer(self.blacklist)
        df = transformer.transform(self.test_df)
        self.assertTrue(self._term_occurs(1997, df))
    
    def test_blacklist_transformer_embs(self):
        '''
        Check if the embeddings related to the token exists after pruning
        '''
        test_blacklist = [1997]
        for i in range(len(self.test_df)):
            toks = self.test_df.iloc[i].doc_toks
            embs = self.test_df.iloc[i].doc_embs
            for i, tok in enumerate(toks):
                if tok == 1997: emb = torch.clone(embs[i]) # deep copy
        transformer = blacklisted_tokens_transformer(test_blacklist)
        df = transformer.transform(self.test_df)
        self.assertFalse(self._embs_occurs(emb, df))
        
    def test_blacklist_transformer_embs_2(self):
        '''
        Check if the two embeddings related to the tokens exist after pruning
        '''
        test_blacklist = [1997, 3280]
        for i in range(len(self.test_df)):
            toks = self.test_df.iloc[i].doc_toks
            embs = self.test_df.iloc[i].doc_embs
            for i, tok in enumerate(toks):
                if tok == 1997: emb_1 = torch.clone(embs[i]) # deep copy
                if tok == 3280: emb_2 = torch.clone(embs[i]) # deep copy
        transformer = blacklisted_tokens_transformer(test_blacklist)
        df = transformer.transform(self.test_df)
        self.assertFalse(self._embs_occurs(emb_1, df) or self._embs_occurs(emb_2, df))

    def test_blacklist_transformer_embs_3(self):
        '''
        Check if the two embeddings related to the tokens exist after pruning
        '''
        embeddings = torch.zeros((len(self.blacklist), 128))
        for i in range(len(self.test_df)):
            toks = self.test_df.iloc[i].doc_toks
            embs = self.test_df.iloc[i].doc_embs
            for i, tok in enumerate(toks):
                if tok in embeddings:
                    embeddings[i, :] = torch.clone(embs[i])
        transformer = blacklisted_tokens_transformer(self.blacklist)
        df = transformer.transform(self.test_df)
        for i in range(embeddings.size()[0]):
            if self._embs_occurs(embeddings[i, :], df) and not torch.equal(embeddings[i, :], torch.zeros(128)):
                self.assertTrue(False)
        self.assertTrue(True)
        
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
        for occurrence in occurrences_tid:
            if occurrence > 0: return True
        return False
    
    def _embs_occurs(self, emb, df):
        '''
        Return true if the embedding occurs in the df
        '''
        if emb.is_cuda: emb = emb.cpu()
        occurrences_emb = []
        for i in range(len(df)):
            n_occurrences = 0
            doc_embs = df.iloc[i].doc_embs
            for element in doc_embs:
                if element.is_cuda: element = element.cpu()
                if torch.equal(emb, element):
                    n_occurrences += 1
            occurrences_emb.append(n_occurrences)
        for occurrence in occurrences_emb:
            if occurrence > 0: return True
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