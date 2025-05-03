import glob
import random
import os

conll_files = glob.glob("../data/conll/*.conll")
all_sents = []
for conll_file in conll_files:
    with open(conll_file) as f:
        sents = f.read().split("\n\n")
    all_sents.extend(sents)

xfolds_langid = {i: [] for i in range(10)}
xfolds_norm = {i: [] for i in range(10)}

random.shuffle(all_sents)
fold_size = len(all_sents) // 10
for i in range(10):
    for j in range(i * fold_size, (i + 1) * fold_size):
        sent = all_sents[j]
        if not sent:
            continue
        src = sent.split("\n")[2].split(" = ")[1]
        tgt = sent.split("\n")[3].split(" = ")[1]
    
        xfolds_langid[i].append(all_sents[j])
        xfolds_norm[i].append((src, tgt))


#
#  Update code below (check)
#
norm_dir = "../data/normalization"
langid_dir = "../data/langid"
for dir in [norm_dir, langid_dir]:
    if not os.path.exists(dir):
        os.makedirs(dir)

for i in range(10):
    train_langid = []
    train_norm = []
    test_langid = xfolds_langid[i]
    test_norm = xfolds_norm[i]

    for j in range(10):
        if j != i:
            train_langid.extend(xfolds_langid[j])
            train_norm.extend(xfolds_norm[j])

    # Write train and test data for langid
    os.makedirs(langid_dir, exist_ok=True)
    with open(os.path.join(langid_dir, f"train_fold_{i}.conll"), "w") as train_file:
        train_file.write("\n\n".join(train_langid))
    with open(os.path.join(langid_dir, f"test_fold_{i}.conll"), "w") as test_file:
        test_file.write("\n\n".join(test_langid))

    # Write train and test data for norm
    os.makedirs(norm_dir, exist_ok=True)
    with open(os.path.join(norm_dir, f"train_fold_{i}.tsv"), "w") as train_file:
        train_file.write("src\ttgt\n")
        for src, tgt in train_norm:
            train_file.write(f"{src}\t{tgt}")
    with open(os.path.join(norm_dir, f"test_fold_{i}.tsv"), "w") as test_file:
        for src, tgt in test_norm:
            test_file.write(f"{src}\t{tgt}")
