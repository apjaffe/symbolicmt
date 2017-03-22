import pywrapfst as fst
import sys
import multiprocessing

def init():
  global tm
  global lm
  global isym
  global osym
  tm = fst.Fst.read(sys.argv[1])
  lm = fst.Fst.read(sys.argv[2])

  isym = {}
  with open(sys.argv[3], "r") as isymfile:
    for line in isymfile:
      x, y = line.strip().split()
      isym[x] = int(y)

  osym = {}
  with open(sys.argv[4], "r") as osymfile:
    for line in osymfile:
      x, y = line.strip().split()
      osym[int(y)] = x

num_threads = 1
if len(sys.argv) > 5:
  num_threads = int(sys.argv[5])


def process_line(line):
  global isym
  global osym
  global tm
  global lm
  # Read input 
  compiler = fst.Compiler()
  arr = line.strip().split() + ["</s>"]
  unks = []
  for i, x in enumerate(arr):
    if x not in isym:
      unks.append(x)
    xsym = isym[x] if x in isym else isym["<unk>"]
    print >> compiler, "%d %d %s %s" % (i, i+1, xsym, xsym)
  print >> compiler, "%s" % (len(arr))
  ifst = compiler.compile()

  # Create the search graph and do search
  graph = fst.compose(ifst, tm)
  graph = fst.compose(graph, lm)
  graph = fst.shortestpath(graph)

  # Read off the output
  out = []
  unkspot = 0
  for state in graph.states():
    for arc in graph.arcs(state):
      if arc.olabel != 0:
        tok = osym[arc.olabel]
        # unk substitution (original words in same order)
        if unkspot < len(unks) and tok == "<unk>":
          out.append(unks[unkspot])
          unkspot += 1
        else:
          out.append(tok)
  return " ".join(reversed(out[1:]))


def main():
  init()
  pool = multiprocessing.Pool(processes = num_threads)
  lines = list(sys.stdin)
  for result in pool.imap(process_line, lines):
    print(result)

main()
