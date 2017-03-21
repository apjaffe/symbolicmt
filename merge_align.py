import sys
import mt_util
from collections import defaultdict

def merge_dicts(d1, d2):
  res = defaultdict(set)
  for k, v in d1.items():
    res[k] |= v
  for k, v in d2.items():
    res[k] |= v
  return res

def main():
  align_e1, align_f1 = mt_util.read_alignment(sys.argv[1]) #std
  align_f2, align_e2 = mt_util.read_alignment(sys.argv[2]) #inv

  for i in xrange(len(align_e1)):
    a1 = align_e1[i]
    a2 = align_e2[i]
    m = merge_dicts(a1, a2)
    line = []
    for k, st in m.items():
      for v in st:
        line.append("%d-%d" % (k,v))
    print(" ".join(line))

main()
