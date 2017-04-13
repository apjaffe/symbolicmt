head -n $1 $2/test.baseline.en > /tmp/hyp
head -n $1 en-de/test.en-de.low.en > /tmp/ref
perl multi-bleu.perl /tmp/ref < /tmp/hyp

