# -*- encoding: utf-8 -*-
import mt_util
from collections import defaultdict
import math
import argparse
import json
import pickle

# python ibmpseudo.py --train_src en-de/train.en-de.low.de --train_tgt en-de/train.en-de.low.en --max-iter 7
# python ibmpseudo.py --train_src en-de/valid.en-de.low.de --train_tgt en-de/valid.en-de.low.en --max-iter 7


'''
  Alignment: P( E | F) = Σ_θ P( θ, F | E) (Equation 98)
  IBM model 1: P( θ, F | E)
  (1) Initialize θ[i,j] = 1 / (|E| + 1) (i for E and j for F) (Equation 100) 
  (2) Expectation-Maximization (EM)
    [E] C[i,j] =  θ[i,j] / Σ_i θ[i,j] (Equation 110)
    [M] θ[i,j] =  C[i,j] / Σ_j C[i,j] (Equation 107)
  (3) Calculate data likelihood (Equation 106)
'''

NULL = "</S>"
NULL_ID = 2
class IBM():
  def __init__(self, bitext, src_text, tgt_text, max_iter, min_freq):
    self.bitext = bitext
    self.max_iter = max_iter

    self.src_freq = mt_util.word_freq_split(src_text)
    self.tgt_freq = mt_util.word_freq_split(tgt_text)
    self.src_vocab = mt_util.get_vocab(self.src_freq, min_freq)
    self.tgt_vocab = mt_util.get_vocab(self.tgt_freq, min_freq)
    self.src_len = len(self.src_vocab)
    self.tgt_len = len(self.tgt_vocab)
    self.src_sents = len(src_text)
    self.tgt_sents = len(tgt_text)
    self.src_token_to_id, self.src_id_to_token = mt_util.word_ids(self.src_freq, min_freq)
    self.tgt_token_to_id, self.tgt_id_to_token = mt_util.word_ids(self.tgt_freq, min_freq)
    self.src_freq_id = self.id_freqs(self.src_freq, self.src_token_to_id)
    self.tgt_freq_id = self.id_freqs(self.tgt_freq, self.tgt_token_to_id)
    
    
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

    fid = self.src_token_to_id[f]
    eid = self.tgt_token_to_id[e]
    max_prob = 0
    max_e = None
    for alte in self.tgt_vocab:
      aid = self.tgt_token_to_id[alte]
      prob = self.theta.get((fid, aid),0)
      if prob > max_prob:
        max_e = alte
        max_prob = prob
    print("P(%s|%s)=%f; P(%s|%s)=%f (%d)" % (f,e,self.theta[(fid,eid)],f,max_e,max_prob, self.tgt_token_to_id[max_e]))
  
  def train(self):
    epsilon = 1.0
    # (1) Initialize θ[i,j] = 1 / (|E| + 1) (Equation 100)
    #self.theta[ e[i], f[j] ] = TODO
    self.theta = dict()
    #for word1 in self.src_vocab:
    #    for word2 in self.tgt_vocab:
    #      self.theta[(word1, word2)] = 1.0 / (self.tgt_len + 1)
    for sent1, sent2 in self.bitext_id:
      for word1 in sent1:
        len1 = len(sent1)
        len2 = len(sent2)
        for word2 in sent2:
          self.theta[(word1, word2)] = 1.0 / (len1+1)#(self.tgt_len+1) #(len2+1)
        self.theta[(word1, NULL_ID)] = 1.0 / (len1+1)
        #self.theta[(word1, NULL_ID)] = 0.0
        #self.theta[(word1, NULL_ID)] = 1.0 / (self.tgt_len+1) 
    #print(len(self.theta))  
    #self.diagnose("wir","we")
    # http://mt-class.org/jhu/slides/lecture-ibm-model1.pdf
    for iter in range(self.max_iter):
      count = defaultdict(float)
      # (2) [E] C[i,j] = θ[i,j] / Σ_i θ[i,j] (Equation 110)
      #count[e[i], f[j]] = TODO
      sums = defaultdict(float)
      for sent1, sent2 in self.bitext_id:
        for word1 in sent1:
          sumprob = 0.0
          for word2 in sent2:
            prob = self.theta[(word1, word2)]
            sumprob += prob
          sumprob += self.theta[(word1, NULL_ID)]
          for word2 in sent2:
            prob = self.theta[(word1, word2)]
            count[(word1,word2)] += prob / sumprob
            sums[word2] += prob / sumprob
          count[(word1,NULL_ID)] += self.theta[(word1, NULL_ID)] / sumprob
          sums[NULL_ID] += self.theta[(word1, NULL_ID)] / sumprob
      # (2) [M] θ[i,j] =  C[i,j] / Σ_j C[i,j] (Equation 107)
      #self.theta[ e[i], f[j] ] = TODO 
      #for word1 in self.src_vocab:
      #  for word2 in self.tgt_vocab:
      for sent1, sent2 in self.bitext_id:
        for word1 in sent1:
          for word2 in sent2:
            #self.theta[(word1, word2)] = count[(word1, word2)] / self.tgt_freq_id[word2]
            self.theta[(word1, word2)] = count[(word1, word2)] / sums[word2]
            
          #self.theta[(word1, NULL_ID)] = count[(word1, NULL_ID)] /  self.tgt_sents
          self.theta[(word1, NULL_ID)] = count[(word1, NULL_ID)] / sums[NULL_ID]

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
        for word1 in sent1:
          innersum = 0.0
          for word2 in sent2:
            innersum += self.theta[(word1, word2)]
          innersum += self.theta[(word1, NULL_ID)]
          outerprodlog += math.log(innersum)
        ll += outerprodlog + math.log(epsilon) - (len(sent1) * math.log(len(sent2) + 1))
      print("Log Likelihood : %f" % (ll / wordlen))
      self.diagnose("dank","thank")
      self.diagnose("wir","we")
      self.diagnose("und","and")
      self.diagnose("vielen","very")
      self.diagnose("vielen","many")
      self.diagnose(".",".")
    
    # (Optional) save/load model parameters for efficiency
    if len(self.theta) < 300000:
      with open("theta.pkl","w") as th:
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
          prob = self.theta[(wordi, wordj)]
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
      #print("\n\n")
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
  args = parser.parse_args()
  bitext, src_text, tgt_text = mt_util.read_bitext_file(args.train_src, args.train_tgt )
  #print(bitext[0], src_text[0], tgt_text[0])
  # bitext = [ ( ['with', 'vibrant', ..], ['mit', 'hilfe',..] ), ([], []) , ..]
  ibm = IBM(bitext, src_text, tgt_text, max_iter = int(args.max_iter), min_freq = int(args.min_freq))
  ibm.train()
  alignments = ibm.align()
  with open(args.output,"w") as alignf:
    dump_align(alignments, alignf)
    #json.dump(alignments, alignf)

if __name__ == '__main__': main()
