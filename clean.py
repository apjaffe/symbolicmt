import sys
from collections import defaultdict

col = int(sys.argv[1])

wid = defaultdict(lambda: len(wid))

x = wid["<eps>"]
for line in sys.stdin:
  arr = line.strip().split()
  if len(arr) > col:
    sys.stdout.write(line)
