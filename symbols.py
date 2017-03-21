import sys
from collections import defaultdict

col = int(sys.argv[1])

wid = defaultdict(lambda: len(wid))

x = wid["<eps>"]
for line in sys.stdin:
  arr = line.strip().split()
  if len(arr) > col:
    x = wid[arr[col]]

if len(sys.argv) > 3:
  col2 = int(sys.argv[2])
  with open(sys.argv[3]) as extra:
    for line in extra:
      arr = line.strip().split()
      if len(arr) > col2:
        x = wid[arr[col2]]

it = list(wid.items())
for x, y in sorted(it, key=lambda x: x[1]):
  print("%s %s" % (x, y))
