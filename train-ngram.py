from __future__ import print_function
import sys
import math
from collections import defaultdict

ctxts1 = 0.0
ctxts2 = defaultdict(lambda: 0.0)
count1 = defaultdict(lambda: 0.0)
count2 = defaultdict(lambda: 0.0)
with open(sys.argv[1], "r") as infile:
  for line in infile:
    vals = line.strip().split() + ["</s>"]
    ctxt = "<s>"
    for val in vals:
      ctxts1 += 1
      ctxts2[ctxt] += 1
      count1[val] += 1
      count2[(ctxt,val)] += 1
      ctxt = val
  
  
if len(sys.argv) > 3:
  with open(sys.argv[3]) as extra:
    col2 = 3
    for line in extra:
      arr = line.strip().split()
      if len(arr) > col2:
        word = arr[col2]
        if word not in count1:
          count1[word] = 0.01

ALPHA_1 = 0.1
ALPHA_UNK = 0.01
ALPHA_2 = 1.0 - ALPHA_1 - ALPHA_UNK
PROB_UNK = ALPHA_UNK / 10000000

stateid = defaultdict(lambda: len(stateid))

with open(sys.argv[2], "w") as outfile:

  # Print the fallbacks
  print("%d %d <eps> <eps> %.4f" % (stateid["<s>"], stateid[""], -math.log(ALPHA_1)), file=outfile)
  for ctxt, val in ctxts2.items():
    if ctxt != "<s>":
      print("%d %d <eps> <eps> %.4f" % (stateid[ctxt], stateid[""], -math.log(ALPHA_1)), file=outfile)
  
  # Print the unigrams
  for word, val in count1.items():
    v1 = val/ctxts1
    print("%d %d %s %s %.4f" % (stateid[""], stateid[word], word, word, -math.log(v1)), file=outfile)
  print("%d %d <unk> <unk> %.4f" % (stateid[""], stateid[""], -math.log(PROB_UNK)), file=outfile)
  
  # Print the bigrams
  for (ctxt, word), val in count2.items():
    v1 = count1[word]/ctxts1
    v2 = val/ctxts2[ctxt]
    val = ALPHA_2 * v2 + ALPHA_1 * v1 + PROB_UNK
    print("%d %d %s %s %.4f" % (stateid[ctxt], stateid[word], word, word, -math.log(val)), file=outfile)
 
	# extra unigrams
#  if len(sys.argv) > 3:
#    with open(sys.argv[3]) as extra:
#      col2 = 3
#      for line in extra:
#        arr = line.strip().split()
#        if len(arr) > col2:
#          word = arr[col2]
#          print("%d %d %s %s %.4f" % (stateid[""], stateid[word], word, word, -math.log(PROB_UNK)), file=outfile)
        	

  
  # Print the final state
  print(stateid["</s>"], file=outfile) 
  
