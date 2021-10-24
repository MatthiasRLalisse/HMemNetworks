
output_dir = 'final'
with open('entity2id.txt', 'r') as f:
  l = [ line.split('\t') for line in f.readlines()[1:] ]
  e2idx = { s: int(i) for s, i in l }

#with open('train2id.txt', 'r') as f:
#  l = [ line.split() for line in f.readlines()[1:] ]
#  train = [ [int(e1), int(e2), int(r)] for e1, e2, r in l ]

with open('relation2id.txt', 'r') as f:
  l = [ line.split('\t') for line in f.readlines()[1:] ]
  r2idx_ = { s: int(i) for s, i in l }


r2idx = r2idx_.copy(); r2invidx = {}; r2inv = {}
#augment r2idx with inverse relations
for r, idx in r2idx_.items():
  r2idx[r+'_r'] = idx + len(r2idx_)
  r2invidx[idx] = idx+len(r2idx_)
  r2inv[r] = r+'_r'

idx2e = { i: e for (e,i) in e2idx.items() }
idx2r = { i: r for (r,i) in r2idx.items() }

print(idx2e)

with open('train2id.txt', 'r') as f:
  l = [ line.split() for line in f.readlines()[1:] ]
  train = [ idx2e[int(e1)]+' '+idx2r[int(r)]+' '+idx2e[int(e2)]+'\n' \
            for e1, e2, r in l ]

train[-1] = train[-1][:-1]
with open('train.txt', 'w') as f: f.writelines(train) 

with open('valid2id.txt', 'r') as f:
  l = [ line.split() for line in f.readlines()[1:] ]
  valid = [ idx2e[int(e1)]+' '+idx2r[int(r)]+' '+idx2e[int(e2)]+'\n' \
            for e1, e2, r in l ]

valid[-1] = valid[-1][:-1]
with open('valid.txt', 'w') as f: f.writelines(train) 

with open('test2id.txt', 'r') as f:
  l = [ line.split() for line in f.readlines()[1:] ]
  test = [ idx2e[int(e1)]+' '+idx2r[int(r)]+' '+idx2e[int(e2)]+'\n' \
            for e1, e2, r in l ]

test[-1] = test[-1][:-1]
with open('test.txt', 'w') as f: f.writelines(train) 


#augment train with inverse relations
with open('train.txt', 'r') as f:
  train = [ line.split() for line in f.readlines()[1:] ]
  #train = [ int(e1), int(r), int(e2) for e1, r, e2 in l ]




#augment train with inverse relations
with open('train2id.txt', 'r') as f:
  l = [ line.split() for line in f.readlines()[1:] ]
  train2idx = [ [int(e1), int(e2), int(r)] for e1, e2, r in l ]

train_out = train.copy()
train2idx_out = train2idx.copy()
for (e1, e2, r), (e1_, r_, e2_) in zip(train2idx, train):
  train_out.append([e2_, r2inv[r_], e1_])
  train2idx_out.append([ e2, e1, r2invidx[r] ])

with open('type_constrain.txt', 'r') as f:
  l = f.readlines()[1:]
  tc_out = l.copy()
  left_types = {}
  right_types = {}
  for i, line in enumerate(l):
    r_idx = int(line.split()[0])
    ents = '\t'.join(line.split()[1:])
    if i%2 == 0: 
      left_types[r_idx] = ents; right_types[r_idx + len(r2idx_)] = ents
    else: 
      right_types[r_idx] = ents; left_types[r_idx + len(r2idx_)] = ents

for i, line in enumerate(l):
  r_idx = int(line.split()[0])
  r_inv_idx = r_idx + len(r2idx_)
  n_ents = len(line.split()[2:])
  ents = '\t'.join(line.split()[1:])
  if i%2 == 0:
    newline = str(r_inv_idx)+'\t'+ right_types[r_idx]+'\n'
  else:
    newline = str(r_inv_idx)+'\t'+ left_types[r_idx]+'\n'
  tc_out.append(newline)

tc_out[-1] = tc_out[-1][:-1]

import os
try: os.mkdir(output_dir)
except FileExistsError: pass

with open(os.path.join(output_dir, 'relation2id.txt'), 'w') as f: 
  out = [ str(len(r2idx))+'\n' ] + \
        [ '%s\t%i\n' % (rname, ridx) for rname, ridx in r2idx.items() ]
  out[-1] = out[-1][:-1]
  f.writelines(out)

with open(os.path.join(output_dir, 'train.txt'), 'w') as f:
  out = [ str(len(train_out))+'\n' ] + \
        [ '\t'.join(l)+'\n' for l in train_out ]
  out[-1] = out[-1][:-1]
  f.writelines(out)

with open(os.path.join(output_dir, 'train2id.txt'), 'w') as f:
  out = [ str(len(train2idx_out))+'\n' ] + \
        [ '%i %i %i\n' % (e1, r, e2) for e1, r, e2 in train2idx_out ]
  out[-1] = out[-1][:-1]
  f.writelines(out)

with open(os.path.join(output_dir, 'type_constrain.txt'), 'w') as f:
  f.writelines(tc_out)


