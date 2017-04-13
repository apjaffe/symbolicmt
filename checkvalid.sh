head -n $1 $2/valid.baseline.en > /tmp/hyp
head -n $1 en-de/valid.en-de.low.en > /tmp/ref
perl multi-bleu.perl /tmp/ref < /tmp/hyp

