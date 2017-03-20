# -*- encoding: utf-8 -*-
import mt_util
import sys
import json
import math
from collections import defaultdict
import multiprocessing

'''
Determine whether set of values is quasi-consecutive
  Examples:
    {1, 2, 3, 4, 5, 6} => True
    {4, 2, 3} => True (equivalent to {2, 3, 4})
    {3} => True
    {1, 2, 4} => True if word at position 3 is not aligned to anything, False otherwise
'''
def quasi_consec(tp, aligned_words):
  mn = min(tp)
  mx = max(tp)
  st = set(aligned_words)
  for i in xrange(mn, mx+1):
    if i not in st and i in aligned_words:
      return False
  return True 

def lookup_align(idxs, align):
  result = set()
  for idx in idxs:
    if idx in align:
      result |= align[idx]
  return result

'''
Given an alignment, extract phrases consistent with the alignment
  Input:
    -e_aligned_words: mapping between E-side words (positions) and aligned F-side words (positions)
    -f_aligned_words: mapping between F-side words (positions) and aligned E-side words (positions)
    -e: E sentence
    -f: F sentence
  Return list of extracted phrases
'''
def phrase_extract(e_aligned_words, f_aligned_words, e, f, max_len):
  extracted_phrases = []
  # Loop over all substrings in the E
  for i1 in range(len(e)):
    for i2 in range(i1, len(e)):
      # Get all positions in F that correspond to the substring from i1 to i2 in E (inclusive)
      tp = lookup_align(xrange(i1,i2+1), e_aligned_words)
      if len(tp) != 0 and quasi_consec(tp, f_aligned_words):
        j1 = min(tp) # min TP
        j2 = max(tp) # max TP
        # Get all positions in E that correspond to the substring from j1 to j2 in F (inclusive)
        sp = lookup_align(xrange(j1, j2+1), f_aligned_words)
        if len(sp) != 0 and min(sp) >= i1 and max(sp) <= i2: # Check that all elements in sp fall between i1 and i2 (inclusive)
          e_phrase = e[i1:i2+1]
          f_phrase = f[j1:j2+1]
          if len(e_phrase) <= max_len and len(f_phrase) <= max_len:
            extracted_phrases.append((e_phrase, f_phrase))
          # Extend source phrase by adding unaligned words
          while j1 >= 0 and j1 not in f_aligned_words: # Check that j1 is unaligned
            j_prime = j2
            while j_prime < len(f) and j_prime not in f_aligned_words: # Check that j2 is unaligned
              f_phrase = f[j1:j_prime+1]
              if len(e_phrase) <= max_len and len(f_phrase) <= max_len:
                extracted_phrases.append((e_phrase, f_phrase))
              j_prime += 1
            j1 -= 1

  return extracted_phrases

def calc_probs(phrases):
  count_fe = defaultdict(int)
  count_e = defaultdict(int)
  for ep, fp in phrases:
    eps = " ".join(ep)
    fps = " ".join(fp)
    count_e[eps] += 1
    count_fe[(fps, eps)] += 1

  prob_fe = dict()
  for ((fps, eps), c) in count_fe.items():
    prob_fe[(fps,eps)] = math.log(count_e[eps]) - math.log(c)
  
  return prob_fe

def dump_probs(probs, fname):
  with open(fname, "w") as out:
    for ((fps, eps), p) in probs.items():
      out.write("%s\t%s\t%.4f\n" % (fps, eps, p))



train_src = mt_util.read_file(sys.argv[1])
train_tgt = mt_util.read_file(sys.argv[2])
src_words = mt_util.split_words(train_src)
tgt_words = mt_util.split_words(train_tgt)
alignment_e, alignment_f = mt_util.read_alignment(sys.argv[3])
outf = sys.argv[4]
max_len = int(sys.argv[5]) # max phrase length

def process_i(i):
  return i, phrase_extract(alignment_e[i], alignment_f[i], tgt_words[i], src_words[i], max_len)


NUM_THREADS = 4
def main():
  pool = multiprocessing.Pool(processes = NUM_THREADS)
  all_phrases = []
  for i, phrases in pool.imap(process_i, xrange(min(len(src_words), len(alignment_e)))):
    all_phrases += phrases
    if i%100 == 0:
      print(i)

  probs = calc_probs(all_phrases)
  dump_probs(probs, outf)

main()
