import numpy as np

for name in ["train", "val", "test"]:
    path = f"data/islenmis/X_{name}.npy"
    X = np.load(path)

    # maske sutunlari: 702 pose, 703 sol el, 704 sag el
    empty = X[:, :, 702:705].sum(axis=2) == 0
    before = (np.abs(X[:, :, 0:702]).sum(axis=2) == 0).sum()

    # hiz sifir olmadigi icin bu kareler bos sayilmiyordu 
    X[:, :, 234:702][empty] = 0

    after = (np.abs(X[:, :, 0:702]).sum(axis=2) == 0).sum()
    np.save(path, X)
    print(f"{name}: tamamen bos kare {before} -> {after}")

print("bitti")
