import sys
def main():
  inpf = sys.argv[1]
  outf = sys.argv[2]
  initial_state = 0
  next_state = 1
  cache = dict()
  with open(inpf) as phrases:
    with open(outf, "w") as output:
      src_words = set()
      tgt_words = set()
      for line in phrases:
        if len(line) > 1:
          parts = line.split("\t")
          f = parts[0].split(" ")
          e = parts[1].split(" ")
          src_words |= set(f)
          tgt_words |= set(e)
          cost = float(parts[2])

          if len(f) == 1 and len(e) == 1:
            output.write("%d %d %s %s %.4f\n" % (initial_state, initial_state, f[0], e[0], cost))
            continue

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


      # allow words to be ignored or inserted
#      INSERT_DELETE_COST = 8
#      for src in src_words:
#        output.write("0 0 %s <eps> %.4f\n" % (src, INSERT_DELETE_COST))
#      for tgt in tgt_words:
#        output.write("0 0 <eps> %s %.4f\n" % (tgt, INSERT_DELETE_COST))

      output.write("0 0 </s> </s>\n")
      output.write("0 0 <unk> <unk>\n")

      output.write("0\n")


main()
