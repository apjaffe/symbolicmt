# -*- encoding: utf-8 -*-
import mt_util
from collections import defaultdict
import math
import argparse
import json
import pickle
import os
import scipy.stats


NULL = "</s>"
NULL_ID = 2
align_gauss = scipy.stats.norm(0, 10)
gauss_pdf = align_gauss.pdf
poly = lambda x: (abs(x)+50)**(-0.5)
base = poly(0)
def align_prob(diff, word1 = None, word2 = None, training = False):
  if training:
    return 1
  if word1 == NULL_ID:
    diff = 0
  elif word2 == NULL_ID:
    diff = 0
  elif diff is None:
    diff = 0
  return poly(diff)/base

def iterplus(iterable, plus):
  for x in iterable:
    yield x
  yield plus

'''
  Alignment: P( E | F) = Σ_θ P( θ, F | E) (Equation 98)
  IBM model 1: P( θ, F | E)
  (1) Initialize θ[i,j] = 1 / (|E| + 1) (i for E and j for F) (Equation 100) 
  (2) Expectation-Maximization (EM)
    [E] C[i,j] =  θ[i,j] / Σ_i θ[i,j] (Equation 110)
    [M] θ[i,j] =  C[i,j] / Σ_j C[i,j] (Equation 107)
  (3) Calculate data likelihood (Equation 106)
'''

class IBM():
  def __init__(self, bitext, src_text, tgt_text, max_iter, min_freq, load_tokens):
    self.bitext = bitext
    self.max_iter = max_iter

    self.src_freq = mt_util.word_freq_split(src_text)
    self.tgt_freq = mt_util.word_freq_split(tgt_text)
    self.src_sents = len(src_text)
    self.tgt_sents = len(tgt_text)
    if load_tokens is not None and os.path.isfile("src"+load_tokens):
      self.src_token_to_id = mt_util.defaultify(json.load(open("src"+load_tokens)))
      self.tgt_token_to_id = mt_util.defaultify(json.load(open("tgt"+load_tokens)))
      self.src_id_to_token = mt_util.invert_ids(self.src_token_to_id)
      self.tgt_id_to_token = mt_util.invert_ids(self.tgt_token_to_id)
    else:
      self.src_token_to_id, self.src_id_to_token = mt_util.word_ids(self.src_freq, int(min_freq))
      self.tgt_token_to_id, self.tgt_id_to_token = mt_util.word_ids(self.tgt_freq, int(min_freq))
      json.dump(dict(self.src_token_to_id), open("src"+load_tokens,"w"))
      json.dump(dict(self.tgt_token_to_id), open("tgt"+load_tokens,"w"))

    self.src_vocab = self.src_token_to_id.keys()
    self.tgt_vocab = self.tgt_token_to_id.keys()
    self.src_len = len(self.src_vocab)
    self.tgt_len = len(self.tgt_vocab)
    self.src_freq_id = self.id_freqs(self.src_freq, self.src_token_to_id)
    self.tgt_freq_id = self.id_freqs(self.tgt_freq, self.tgt_token_to_id)
    print("Vocab size: %d / %d" % (self.src_len, self.tgt_len)) 
    
    self.bitext_id = [(self.to_ids(self.src_token_to_id, x), self.to_ids(self.tgt_token_to_id, y)) for x,y in self.bitext]
    
  def to_ids(self, dct, sent):
    return [dct[word] for word in sent]
  
  def id_freqs(self, freqs, lookup):
    result = dict()
    for word, freq in freqs.items():
      result[lookup[word]] = freq
    return result
    
  def diagnose(self, f, e):
    tmp = f
    f = e
    e = tmp
    if f not in self.src_token_to_id:
      return
    if e not in self.tgt_token_to_id:
      return

    fid = self.src_token_to_id[f]
    eid = self.tgt_token_to_id[e]
    if (fid,eid) not in self.theta:
      return
    max_prob = 0
    max_e = None
    for alte in self.tgt_vocab:
      aid = self.tgt_token_to_id[alte]
      prob = self.theta.get((fid, aid),0)
      if prob > max_prob:
        max_e = alte
        max_prob = prob
    print("P(%s|%s)=%f; P(%s|%s)=%f (%d)" % (f,e,self.theta[(fid,eid)],f,max_e,max_prob, self.tgt_token_to_id[max_e]))
  
  def train(self, theta_pkl = "theta.pkl"):
    # (1) Initialize θ[i,j] = 1 / (|E| + 1) (Equation 100)
    #self.theta[ e[i], f[j] ] = ...
    self.theta = dict()
    if os.path.isfile(theta_pkl):
      with open(theta_pkl) as th:
        self.theta = pickle.load(th)
      maxlen = 0
      for sent1, sent2 in self.bitext_id:
        for word1 in iterplus(sent1,NULL_ID):
          len1 = len(sent1)
          len2 = len(sent2)
          maxlen = max(maxlen,len1)
    else:
      maxlen = 0
      for sent1, sent2 in self.bitext_id:
        for word1 in iterplus(sent1,NULL_ID):
          len1 = len(sent1)
          len2 = len(sent2)
          maxlen = max(maxlen,len1)
          for word2 in sent2:
            self.theta[(word1, word2)] = 1.0 / (self.src_len)
    print("Theta size: %d" % len(self.theta))  
    # http://mt-class.org/jhu/slides/lecture-ibm-model1.pdf
    epsilon = 1.0 / maxlen
    for iter in range(self.max_iter):
      count = defaultdict(float)
      # (2) [E] C[i,j] = θ[i,j] / Σ_i θ[i,j] (Equation 110)
      #count[e[i], f[j]] = ...
      sums = defaultdict(float)
      for sent1, sent2 in self.bitext_id:
        for i, word1 in enumerate(iterplus(sent1, NULL_ID)):
          sumprob = 0.0
          for j, word2 in enumerate(sent2):
            prob = self.theta[(word1, word2)] * align_prob(i-j, word1, word2, True)
            sumprob += prob
          for j, word2 in enumerate(sent2):
            prob = self.theta[(word1, word2)] * align_prob(i-j, word1, word2, True)
            count[(word1,word2)] += prob / sumprob
            sums[word2] += prob / sumprob

      # (2) [M] θ[i,j] =  C[i,j] / Σ_j C[i,j] (Equation 107)
      #self.theta[ e[i], f[j] ] = ...
      for sent1, sent2 in self.bitext_id:
        for word1 in iterplus(sent1, NULL_ID):
          for word2 in sent2:
            self.theta[(word1, word2)] = count[(word1, word2)] / sums[word2]
            
      # (3) Calculate log data likelihood (Equation 106)
      ll = 0.0
      wordlen = 0
      for sent1, sent2 in self.bitext_id:
        outerprodlog = 0.0
        wordlen += len(sent1)
        for word1 in iterplus(sent1, NULL_ID):
          innersum = 0.0
          for word2 in sent2:
            innersum += self.theta[(word1, word2)]
          outerprodlog += math.log(innersum)
        ll += outerprodlog + math.log(epsilon) - (len(sent1) * math.log(len(sent2) + 1))
      print("Log Likelihood : %f" % (ll / wordlen))
      self.diagnose("dank","thank")
      self.diagnose("wir","we")
      self.diagnose("und","and")
      self.diagnose("vielen","very")
      self.diagnose("vielen","many")
      self.diagnose(".",".")
      self.diagnose("die","the")
      self.diagnose("das","the")
      self.diagnose("teilen","share")
    
    # Save model parameters for efficiency if practical
    if len(self.theta) < 300000:
      with open(theta_pkl,"w") as th:
        pickle.dump(self.theta, th)

  def tgt_to_tok(self, id):
    if id == NULL_ID:
      return "<eps>"
    else:
      return self.tgt_id_to_token[id]

  def src_to_tok(self, id):
    if id == NULL_ID:
      return "<eps>"
    else:
      return self.src_id_to_token[id]

  def dump_words(self, outf):
    thresh = 0.0000001
    PENALTY = -math.log(0.01)
    for (word1, word2), prob in self.theta.items():
      if prob >= thresh:
        unk_penalty = -math.log(0.9) if word1==NULL_ID else 0
        outf.write("%s\t%s\t%.4f\n" % (self.tgt_to_tok(word2), self.src_to_tok(word1), -math.log(prob) + PENALTY + unk_penalty))

  def align(self):
    alignments = []
    for idx, (f, e) in enumerate(self.bitext_id):
      align = []
      for i in xrange(len(f)):
        wordi = f[i]
        # ARGMAX_j θ[i,j[
        #max_j, max_prob = argmax_j(f, e[i])
        max_prob = 0
        max_j = None
        for j in xrange(len(e) + 1):
          wordj = e[j] if j < len(e) else NULL_ID
          diff = i-j if j < len(e) else None
          prob = self.theta[(wordi, wordj)] * align_prob(i-j, None, None, False)
          if prob > max_prob:
            max_prob = prob 
            max_j = j
        if max_j < len(e):
          align.append((i,max_j))
      alignments.append(align)
      if idx % 1000 == 0:
        print(idx)
    return alignments
  
  def inv_align(self):
    alignments = []
    for idx, (f, e) in enumerate(self.bitext_id):
      align = []
      for i in xrange(len(e)):
        wordi = e[i]
        # ARGMAX_j θ[i,j] or other alignment in Section 11.6 (e.g., Intersection, Union, etc)
        #max_j, max_prob = argmax_j(f, e[i])
        max_prob = 0
        max_j = None
        for j in xrange(len(f)):
          wordj = f[j] if j < len(f) else NULL_ID
          diff = i-j if j < len(f) else None
          prob = self.theta[(wordj, wordi)] * align_prob(i-j, None, None, False)
          if prob > max_prob:
            max_prob = prob 
            max_j = j
        if max_j < len(f):
          align.append((i,max_j))
      alignments.append(align)
      if idx % 1000 == 0:
        print(idx)
    return alignments

def dump_align(alignments, f):    
  for line in alignments:
    out = []
    for i, word in line:
      out.append("%d-%d" % (word, i))
    f.write(" ".join(out)+"\n")

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--train_tgt')
  parser.add_argument('--train_src')
  parser.add_argument('--max_iter', default = 7)
  parser.add_argument('--output')
  parser.add_argument('--min_freq', default = 1)
  parser.add_argument('--model', default = "theta.pkl")
  parser.add_argument('--tokens', default = "tokens.json")
  parser.add_argument('--output2')
  args = parser.parse_args()
  bitext, src_text, tgt_text = mt_util.read_bitext_file(args.train_src, args.train_tgt )
  ibm = IBM(bitext, src_text, tgt_text, max_iter = int(args.max_iter), min_freq = int(args.min_freq), load_tokens = args.tokens)
  ibm.train(args.model)
  alignments = ibm.inv_align()
  with open(args.output,"w") as alignf:
    dump_align(alignments, alignf)
  with open(args.output2,"w") as wordf:
    ibm.dump_words(wordf)

if __name__ == '__main__': main()
