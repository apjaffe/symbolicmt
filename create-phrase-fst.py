import sys
def main():
  inpf = sys.argv[1]
  outf = sys.argv[2]
  initial_state = 0
  next_state = 1
  cache = dict()
  with open(inpf) as phrases:
    with open(outf, "w") as output:
      for line in phrases:
        if len(line) > 1:
          parts = line.split("\t")
          f = parts[0].split(" ")
          e = parts[1].split(" ")
          cost = float(parts[2])
          last_state = initial_state
          sofar = ""
          for wordf in f:
            if (sofar + wordf + " ") not in cache: 
              if sofar in cache:
                last_state = cache[sofar]
              output.write("%d %d %s %s\n" % (last_state, next_state, wordf, "<eps>"))
              sofar += wordf + " "
              cache[sofar] = next_state
              last_state = next_state
              next_state += 1
            else:
              sofar += wordf + " "
          
          for worde in e:
            if (sofar + worde + "~") not in cache: 
              if sofar in cache:
                last_state = cache[sofar]
              
              output.write("%d %d %s %s\n" % (last_state, next_state, "<eps>", worde))
              sofar += worde + "~"
              cache[sofar] = next_state
              last_state = next_state
              next_state += 1
            else:
              sofar += worde + "~"
          
          if sofar in cache:
            last_state = cache[sofar]

          output.write("%d %d <eps> <eps> %.4f\n" % (last_state, initial_state, cost))

      output.write("0 0 </s> </s>\n")
      output.write("0 0 <unk> <unk>\n")
      output.write("0\n")


main()
