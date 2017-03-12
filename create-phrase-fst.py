import sys
def main():
  inpf = sys.argv[1]
  outf = sys.argv[2]
  initial_state = 0
  next_state = 1
  with open(inpf) as phrases:
    with open(outf, "w") as output:
      for line in phrases:
        if len(line) > 1:
          parts = line.split("\t")
          f = parts[0].split(" ")
          e = parts[1].split(" ")
          cost = float(parts[2])
          last_state = initial_state
          for wordf in f:
            output.write("%d %d %s %s\n" % (last_state, next_state, wordf, "<eps>"))
            last_state = next_state
            next_state += 1
          
          for worde in e:
            output.write("%d %d %s %s\n" % (last_state, next_state, "<eps>", worde))
            last_state = next_state
            next_state += 1
          
          output.write("%d %d <eps> <eps> %f\n" % (last_state, initial_state, cost))


main()
