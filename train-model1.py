# -*- encoding: utf-8 -*-
import mt_util
from collections import defaultdict
import math
import argparse
import json
import pickle
import scipy.stats
import os

# python ibmpseudo.py --train_src en-de/train.en-de.low.de --train_tgt en-de/train.en-de.low.en --max-iter 7
# python ibmpseudo.py --train_src en-de/valid.en-de.low.de --train_tgt en-de/valid.en-de.low.en --max-iter 7


NULL = "</S>"
NULL_ID = 2
align_gauss = scipy.stats.norm(0, 10)
gauss_pdf = align_gauss.pdf
poly = lambda x: (abs(x)+20)**(-0.5)
def align_prob(diff, word1 = None, word2 = None):
  if word1 == NULL_ID:
    diff = 0
  elif word2 == NULL_ID:
    diff = 0
  elif diff is None:
    diff = 0
  return poly(diff) 

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
    #self.src_vocab = mt_util.get_vocab(self.src_freq, min_freq)
    #self.tgt_vocab = mt_util.get_vocab(self.tgt_freq, min_freq)
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
    #for k, v in self.theta.items():
    #  if v > 1:
    #    print("EXCEEDS 1: P(%s|%s)=%f" % (self.src_id_to_token[k[0]], self.tgt_id_to_token[k[1]], v))
    #    break
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
    #self.theta[ e[i], f[j] ] = TODO
    self.theta = dict()
    if os.path.isfile(theta_pkl):
      with open(theta_pkl) as th:
        self.theta = pickle.load(th)
    #for word1 in self.src_vocab:
    #    for word2 in self.tgt_vocab:
    #      self.theta[(word1, word2)] = 1.0 / (self.tgt_len + 1)
    maxlen = 0
    for sent1, sent2 in self.bitext_id:
      for word1 in iterplus(sent1,NULL_ID):
        len1 = len(sent1)
        len2 = len(sent2)
        maxlen = max(maxlen,len1)
        for word2 in sent2:
          self.theta[(word1, word2)] = 1.0 / (self.src_len)#(len1+1)#(self.tgt_len+1) #(len2+1)
        #self.theta[(word1, NULL_ID)] = 1.0 / (self.tgt_len)#(len1+1)
        #self.theta[(word1, NULL_ID)] = 0.0
        #self.theta[(word1, NULL_ID)] = 1.0 / (self.tgt_len+1) 
    print("Theta size: %d" % len(self.theta))  
    #self.diagnose("wir","we")
    # http://mt-class.org/jhu/slides/lecture-ibm-model1.pdf
    epsilon = 1.0 / maxlen
    for iter in range(self.max_iter):
      count = defaultdict(float)
      # (2) [E] C[i,j] = θ[i,j] / Σ_i θ[i,j] (Equation 110)
      #count[e[i], f[j]] = TODO
      sums = defaultdict(float)
      for sent1, sent2 in self.bitext_id:
        for i, word1 in enumerate(iterplus(sent1, NULL_ID)):
          sumprob = 0.0
          for j, word2 in enumerate(sent2):
            prob = self.theta[(word1, word2)] * align_prob(i-j, word1, word2)
            sumprob += prob
          #sumprob += self.theta[(word1, NULL_ID)]
          for j, word2 in enumerate(sent2):
            prob = self.theta[(word1, word2)] * align_prob(i-j, word1, word2)
            count[(word1,word2)] += prob / sumprob
            sums[word2] += prob / sumprob
          #count[(word1,NULL_ID)] += self.theta[(word1, NULL_ID)] * align_prob(None) / sumprob
          #sums[NULL_ID] += self.theta[(word1, NULL_ID)] * align_prob(None) / sumprob
      # (2) [M] θ[i,j] =  C[i,j] / Σ_j C[i,j] (Equation 107)
      #self.theta[ e[i], f[j] ] = TODO 
      #for word1 in self.src_vocab:
      #  for word2 in self.tgt_vocab:
      for sent1, sent2 in self.bitext_id:
        for word1 in iterplus(sent1, NULL_ID):
          for word2 in sent2:
            #self.theta[(word1, word2)] = count[(word1, word2)] / self.tgt_freq_id[word2]
            self.theta[(word1, word2)] = count[(word1, word2)] / sums[word2]
            
          #self.theta[(word1, NULL_ID)] = count[(word1, NULL_ID)] /  self.tgt_sents
          #self.theta[(word1, NULL_ID)] = count[(word1, NULL_ID)] / sums[NULL_ID]

      #print(len(self.theta))      
      #self.diagnose("wir","we")
 
      #with open("theta.pkl","w") as thetaf:
      #  pickle.dump(self.theta, thetaf)
      
      #print("pickled")
      
      # (3) Calculate log data likelihood (Equation 106)
      #ll = TODO
      ll = 0.0
      wordlen = 0
      for sent1, sent2 in self.bitext_id:
        outerprodlog = 0.0
        wordlen += len(sent1)
        for word1 in iterplus(sent1, NULL_ID):
          innersum = 0.0
          for word2 in sent2:
            innersum += self.theta[(word1, word2)]
          #innersum += self.theta[(word1, NULL_ID)]
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
    
    # (Optional) save/load model parameters for efficiency
    if len(self.theta) < 300000:
      with open(theta_pkl,"w") as th:
        pickle.dump(self.theta, th)
		#[0] Log Likelihood : -5.232084
		#[1] Log Likelihood : -4.542094
		#[2] Log Likelihood : -4.321830
		#[3] Log Likelihood : -4.244019
		#[4] Log Likelihood : -4.209469
		#[5] Log Likelihood : -4.191590
		#[6] Log Likelihood : -4.181324

  def align(self):
    alignments = []
    for idx, (f, e) in enumerate(self.bitext_id):
      align = []
      for i in xrange(len(f)):
        wordi = f[i]
        max_prob = 0
        max_j = None
        for j in xrange(len(e) + 1):
          wordj = e[j] if j < len(e) else NULL_ID
          diff = i-j if j < len(e) else None
          prob = self.theta[(wordi, wordj)] * align_prob(i-j)
          #print("P(%s|%s)=%f"%(self.src_id_to_token[wordi], self.tgt_id_to_token[wordj], prob))
          if prob > max_prob:
            max_prob = prob 
            max_j = j
        if max_j < len(e):
          align.append((i,max_j))
        # ARGMAX_j θ[i,j] or other alignment in Section 11.6 (e.g., Intersection, Union, etc)
        #max_j, max_prob = argmax_j(f, e[i])
      #self.plot_alignment((max_j, max_prob), e, f)
      alignments.append(align)
      if idx % 100 == 0:
        print(idx)
    return alignments
  
  def inv_align(self):
    alignments = []
    for idx, (f, e) in enumerate(self.bitext_id):
      align = []
      for i in xrange(len(e)):
        wordi = e[i]
        max_prob = 0
        max_j = None
        for j in xrange(len(f)+1):
          wordj = f[j] if j < len(f) else NULL_ID
          diff = i-j if j < len(f) else None
          prob = self.theta[(wordj, wordi)] * align_prob(i-j)
          #print("P(%s|%s)=%f"%(self.src_id_to_token[wordi], self.tgt_id_to_token[wordj], prob))
          if prob > max_prob:
            max_prob = prob 
            max_j = j
        if max_j < len(f):
          align.append((i,max_j))
        # ARGMAX_j θ[i,j] or other alignment in Section 11.6 (e.g., Intersection, Union, etc)
        #max_j, max_prob = argmax_j(f, e[i])
      #self.plot_alignment((max_j, max_prob), e, f)
      alignments.append(align)
      if idx % 100 == 0:
        print(idx)
    return alignments

#def argmax(arr):
#  return max(xrange(len(arr)), key = lambda i: arr[i])

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
  args = parser.parse_args()
  bitext, src_text, tgt_text = mt_util.read_bitext_file(args.train_src, args.train_tgt )
  #print(bitext[0], src_text[0], tgt_text[0])
  # bitext = [ ( ['with', 'vibrant', ..], ['mit', 'hilfe',..] ), ([], []) , ..]
  ibm = IBM(bitext, src_text, tgt_text, max_iter = int(args.max_iter), min_freq = int(args.min_freq), load_tokens = args.tokens)
  ibm.train(args.model)
  alignments = ibm.inv_align()
  with open(args.output,"w") as alignf:
    dump_align(alignments, alignf)
    #json.dump(alignments, alignf)

if __name__ == '__main__': main()
