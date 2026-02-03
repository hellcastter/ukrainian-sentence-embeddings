import random

max_number = 200_000
SRC_FILE = "local_datasets/ubertext_triplets.csv"
DST_FILE = "local_datasets/ubertext_triplets_200K.csv"

with open(SRC_FILE, "r", encoding="utf-8") as src_f:
    header = src_f.readline()
    rows = src_f.readlines()

random.shuffle(rows)
subset = rows[:max_number]

with open(DST_FILE, "w", encoding="utf-8") as dst_f:
    dst_f.write(header)
    dst_f.writelines(subset)
