#!/bin/bash
set -e

##### NOTE
# This assumes that you have OpenFST and the Python bindings installed

DATA_DIR=en-de
SCRIPT_DIR=.
#OUT_DIR=output-full2
TRAIN_DATA=en-de/valid.en-de.low
OUT_DIR=output-tmp4
#TRAIN_DATA=en-de/train.en-de.low.filt
mkdir -p $OUT_DIR

# *** Train n-gram language model and create an FST
python $SCRIPT_DIR/train-ngram.py $TRAIN_DATA.en $OUT_DIR/ngram-fst.txt

# *** Implement 1: Train IBM Model 1 and find alignment
#python $SCRIPT_DIR/train-model1.py --train_src $TRAIN_DATA.de --train_tgt $TRAIN_DATA.en --output $OUT_DIR/alignment.txt --max_iter 50
python $SCRIPT_DIR/train-model1.py --train_src $TRAIN_DATA.en --train_tgt $TRAIN_DATA.de --output $OUT_DIR/alignment.txt --max_iter 20 --min_freq 1 --model theta_2v0.pkl --tokens tokens_1v.json --output2 $OUT_DIR/word.txt
#python $SCRIPT_DIR/train-model1.py --train_src $TRAIN_DATA.en --train_tgt $TRAIN_DATA.de --output $OUT_DIR/alignment.txt --max_iter 15 --min_freq 1 --model theta_5.pkl --tokens tokens_1b.json --output2 $OUT_DIR/word.txt

#python $SCRIPT_DIR/train-model1.py --train_src $TRAIN_DATA.en --train_tgt $TRAIN_DATA.de --output $OUT_DIR/alignmentinv.txt


# *** Implement 2: Extract and score phrases
python $SCRIPT_DIR/phrase-extract.py $TRAIN_DATA.de $TRAIN_DATA.en $OUT_DIR/alignment.txt $OUT_DIR/phrase.txt 3

cat $OUT_DIR/word.txt $OUT_DIR/phrase.txt > $OUT_DIR/combo.txt

# *** Implement 3: Create WFSTs for phrases
python $SCRIPT_DIR/create-phrase-fst.py $OUT_DIR/combo.txt $OUT_DIR/phrase-fst.txt

# *** Compile WFSTs into a single model
python $SCRIPT_DIR/symbols.py 2 < $OUT_DIR/phrase-fst.txt > $OUT_DIR/phrase-fst.isym
python $SCRIPT_DIR/symbols.py 2 < $OUT_DIR/ngram-fst.txt > $OUT_DIR/ngram-fst.isym
fstcompile --isymbols=$OUT_DIR/ngram-fst.isym --osymbols=$OUT_DIR/ngram-fst.isym $OUT_DIR/ngram-fst.txt | fstarcsort > $OUT_DIR/ngram-fst.fst
fstcompile --isymbols=$OUT_DIR/phrase-fst.isym --osymbols=$OUT_DIR/ngram-fst.isym $OUT_DIR/phrase-fst.txt | fstarcsort > $OUT_DIR/phrase-fst.fst

# *** Normally we could do this for efficiency purposes, but it takes a lot of memory, so we keep the FSTs separate
# fstcompose $OUT_DIR/phrase-fst.fst $OUT_DIR/ngram-fst.fst | fstarcsort > $OUT_DIR/tm-fst.fst

# *** Compose and find the best path for each WFST
for f in mini valid test blind; do
  echo "python $SCRIPT_DIR/decode.py $OUT_DIR/phrase-fst.fst $OUT_DIR/ngram-fst.fst $OUT_DIR/phrase-fst.isym $OUT_DIR/ngram-fst.isym < $DATA_DIR/$f.en-de.low.de > $OUT_DIR/$f.baseline.en"
  python $SCRIPT_DIR/decode.py $OUT_DIR/phrase-fst.fst $OUT_DIR/ngram-fst.fst $OUT_DIR/phrase-fst.isym $OUT_DIR/ngram-fst.isym 8 < $DATA_DIR/$f.en-de.low.de > $OUT_DIR/$f.baseline.en
  if [[ -e $DATA_DIR/$f.en-de.low.en ]]; then
    perl $SCRIPT_DIR/multi-bleu.perl $DATA_DIR/$f.en-de.low.en < $OUT_DIR/$f.baseline.en
  fi
done
