from __future__ import print_function
import sys
import random
import math
import dynet as dy
from collections import defaultdict

# Define the hyperparameters
N = 2
EVAL_EVERY = 50000
EMB_SIZE = 64
HID_SIZE = 64

word_frequencies = defaultdict(int)

train_file = "en-de/train.en-de.low.en"
if len(sys.argv) > 1:
  train_file = sys.argv[1]
valid_file = "en-de/valid.en-de.low.en"
if len(sys.argv) > 2:
  valid_file = sys.argv[2]

# Loop over the words, counting the frequency
# open the training file
# count the frequency of words
with open(train_file) as train:
  lines = train.read().split("\n")
  for line in lines:
    #word_frequencies["<s>"] += 1
    #word_frequencies["</s>"] += 1
    for word in line.split(" "):
      word_frequencies[word] += 1


# Create the word vocabulary for all words > 1
# loop over the words, and create word IDs for all words < 1
wids = defaultdict(lambda: 0)
wids["<unk>"] = 0
wids["<s>"] = 1
wids["</s>"] = 2
uwid = wids["<unk>"]
swid = wids["<s>"]
stwid = wids["</s>"]
word_lookup = dict()
for word, freq in word_frequencies.items():
  if freq > 12:
    wids[word] = len(wids) 

for word, wid in wids.items():
  word_lookup[wid] = word

def get_wid(word):
  if word in wids:
    return wids[word]
  else:
    return wids["<unk>"]

def feature_function(words, i):
  features = []
  #if i > 1:
  #  features.append(get_wid(words[i-2]))
  #else:
  #  features.append(swid)
  
  if i > 0:
    features.append(get_wid(words[i-1]))
  else:
    features.append(swid)
  
  return features

# Read in the training and validation data and parse it into context/word pairs
def create_data(fname):
  pairs = []
  with open(fname) as train:
    lines = train.read().split("\n")
    for line in lines:
      words = line.split(" ")
      words.append("</s>")
      for i in xrange(len(words)):
        pairs.append((feature_function(words, i), get_wid(words[i])))

  return pairs

train_data = create_data(train_file)
valid_data = create_data(valid_file)

# Create the neural network model including lookup parameters, etc
model = dy.Model()
M_p = model.add_lookup_parameters((len(wids), EMB_SIZE))
W_mh_p = model.add_parameters((HID_SIZE, EMB_SIZE * (N-1)))
b_h_p = model.add_parameters((HID_SIZE))
W_hs_p = model.add_parameters((len(wids), HID_SIZE))
b_s_p = model.add_parameters((len(wids)))
trainer = dy.SimpleSGDTrainer(model)

params = [M_p, W_mh_p, b_h_p, W_hs_p, b_s_p]

# The function to calculate the prediction
def calc_function(features):
  dy.renew_cg()
  m_vals = [dy.lookup(M_p,feature) for feature in features]
  m_val = dy.concatenate(m_vals)
  W_mh = dy.parameter(W_mh_p)
  b_h = dy.parameter(b_h_p)
  W_hs = dy.parameter(W_hs_p)
  b_s = dy.parameter(b_s_p)
  h_val = dy.tanh(W_mh * m_val + b_h)
  s_val = W_hs * h_val + b_s
  return s_val

def calculate_perplexity(data):
  log_loss = 0
  count = 0
  for ctx, wid in data:
    s_val = calc_function(ctx)
    log_loss += dy.pickneglogsoftmax(s_val, wid).value()
    if wid == uwid:
      log_loss += math.log(10000000) 
    if wid != stwid:
      count += 1
  return math.exp(log_loss / count)

def generate_sentence():
  ctx = [swid]
  next_wid = -1
  out = []
  while next_wid != stwid:
    s_val = calc_function(ctx)
    p = dy.softmax(s_val).value()
    #print(len(p))
    #print(sum(p))
    p[0] = 0 # never select <unk>
    rnd = random.random() * sum(p)
    i = 0
    lenp = len(p)
    while rnd > 0 and i < lenp:
      rnd -= p[i]
      i+=1
    next_wid = i-1
    out.append(word_lookup[next_wid])
    #ctx[0] = ctx[1]
    #ctx[1] = next_wid
    ctx[0] = next_wid
  return out

def write_ngrams(f):
  stateid = defaultdict(lambda: len(stateid))
  stateid["<s>"] = 0
  err_prob = 0.1
  base_prob = 0.001 #1/len(wids)
  with open(f,"w") as outfile:
    #bigrams
    for word1, wid1 in wids.items():
      ctx = [wid1]
      s_val = calc_function(ctx)
      p = dy.softmax(s_val).value()
      for word2, wid2 in wids.items():
        if p[wid2] > base_prob:
          print("%d %d %s %s %.4f" % (stateid[wid1], stateid[wid2], word2, word2, -math.log(p[wid2])), file = outfile)

    #fallbacks
    for word1, wid1 in wids.items():
      print("%d %d <eps> <eps> %.4f" % (stateid[wid1], stateid[""], -math.log(err_prob)), file=outfile)

    #unigrams
    b_s = dy.parameter(b_s_p)
    b_val = dy.softmax(b_s).value()
    for word1, wid1 in wids.items():
      print("%d %d %s %s %.4f" % (stateid[""], stateid[wid1], word1, word1, -math.log(b_val[wid1])),file=outfile)

    #final state
    print(stateid["</s>"], file=outfile)


if True:
  (M_p, W_mh_p, b_h_p, W_hs_p, b_s_p) = model.load("model1.mdl")
  print(generate_sentence())

# For data in the training set, train. Evaluate dev set occasionally
# basically the same as the log-linear model
NUM_EPOCHS = 10
for epoch in xrange(NUM_EPOCHS):
  epoch_loss = 0
  random.shuffle(train_data)
  i = 0
  lnt = len(train_data)
  for ctx, wid in train_data:
    s_val = calc_function(ctx)
    loss = dy.pickneglogsoftmax(s_val, wid)
    epoch_loss += loss.value()
    loss.backward()
    trainer.update()
    i += 1
    if i % EVAL_EVERY == 0:
      perplexity = calculate_perplexity(valid_data)
      print("Epoch=%d, spot=%d, total=%d, loss=%f, validation perplexity=%f" % (epoch, i, lnt, epoch_loss, perplexity))
      print(" ".join(generate_sentence()))
      model.save("model1.mdl",params)
      print("model saved")
      write_ngrams("output-neural/ngram-fst.txt")
      print("ngrams written")
      


