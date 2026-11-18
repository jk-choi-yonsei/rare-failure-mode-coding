"""Fast pre-flight: exercise every fold's inner split without training.

Catches split-construction bugs in seconds instead of an hour into a run.
"""
from encoder_finetune import inner_split, load_task

for ds in ("fmc", "maintie_m1"):
    ids, texts, labels, folds, classes = load_task(ds)
    c2i = {c: i for i, c in enumerate(classes)}
    y = [c2i[l] for l in labels]
    print(f"== {ds}: n={len(ids)} classes={len(classes)}")
    for seed in (42, 43, 44):
        for f in range(5):
            tr_i = [i for i, fl in enumerate(folds) if fl != f]
            te_i = [i for i, fl in enumerate(folds) if fl == f]
            trr, var = inner_split(tr_i, y, seed)
            assert not (set(trr) & set(var)), "train/val overlap"
            assert set(trr) | set(var) == set(tr_i), "lost records"
            assert not (set(trr) & set(te_i)), "train/test leak"
            assert len(var) > 0, "empty validation"
        print(f"   seed {seed}: 5 folds OK "
              f"(last fold train={len(trr)} val={len(var)} test={len(te_i)})")
print("ALL SPLIT CHECKS PASSED")
