from collections import defaultdict
def word_freqs(lines):
  word_frequencies = defaultdict(int)
  for line in lines:
    for word in line.split(" "):
      word_frequencies[word] += 1
  return word_frequencies
  
def word_freq_split(lines):
  word_frequencies = defaultdict(int)
  for line in lines:
    for word in line:
      word_frequencies[word] += 1
  return word_frequencies
  
def get_vocab(freqs):
  vocab = list()
  for word, freq in freqs.items():
    vocab.append(word)
  return vocab

def defaultify(dct):
  wids = defaultdict(lambda: 0)
  for k, v in dct.items():
    wids[k] = v
  return wids

def word_ids(word_frequencies, min_freq=2):
  wids = defaultdict(lambda: 0)
  wids["<UNK>"] = 0
  wids["<S>"] = 1
  wids["</S>"] = 2
  for word, freq in word_frequencies.items():
    if freq >= min_freq:
      wids[word] = len(wids) 

  return wids, invert_ids(wids)

def invert_ids(wids):
  word_lookup = defaultdict(lambda: "<UNK>")
  for word, wid in wids.items():
    word_lookup[wid] = word
  
  return word_lookup

def get_get_wid(wids):
  def get_wid(wid):
    if word in wids:
      return wids[word]
    else:
      return wids["<UNK>"]
   
def clean(wd):
  return wd.decode("utf-8").encode("ascii","ignore")
    
def make_batches(training, batch_size, min_len = 3):
  filtered = filter(lambda x: len(x[0]) >= min_len and len(x[1]) >= min_len, training)

  filtered.sort(key=lambda x: -len(x[0]))
  batches = []
  current_batch = []
  batch_len = None
  for pair in filtered:
    if batch_len is None:
      batch_len = len(pair[0])
      current_batch = [pair]
    elif len(pair[0]) != batch_len or len(current_batch) == batch_size:
      batches.append(current_batch)
      batch_len = len(pair[0])
      current_batch = [pair]
    else:
      current_batch.append(pair)
  if len(current_batch) > 0:
    batches.append(current_batch)
  return batches
  
def read_file(fname):
    with open(fname) as f:
      return f.read().split("\n")

def split_words(lines):
  return [line.split(" ") for line in lines]
      
def read_bitext_file(f1, f2):
  lines1 = read_file(f1)
  lines2 = read_file(f2)
  words1 = split_words(lines1)
  words2 = split_words(lines2)
  return zip(words1, words2), words1, words2 
