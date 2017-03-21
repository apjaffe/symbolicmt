import sys
from collections import defaultdict

col = int(sys.argv[1])

print("0 5223 <eps> <eps>")

ns = 0
words = set()
for line in sys.stdin:
  arr = line.strip().split()
  if len(arr) > col:
    words.add(arr[2])
    ns = max(int(arr[0]),ns)
    ns = max(int(arr[1]),ns)
    sys.stdout.write(line)

if len(sys.argv) > 2:
  with open(sys.argv[2]) as extra:
    col2 = 3
    for line in extra:
      arr = line.strip().split()
      if len(arr) > col2:
        word = arr[col2]
        if word not in words:
          words.add(word)
          sys.stdout.write("%d %d %s %s %.4f \n" % (ns+1, ns+1, word, word, 100))

sys.stdout.write("7223 4 </s> </s> 5.6292\n")
sys.stdout.write("4\n")

