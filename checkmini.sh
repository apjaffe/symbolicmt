head -n $1 $2/mini.baseline.en > /tmp/hyp
head -n $1 en-de/mini.en-de.low.en > /tmp/ref
perl multi-bleu.perl /tmp/ref < /tmp/hyp

