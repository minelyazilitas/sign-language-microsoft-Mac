import os
import glob
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))

p = argparse.ArgumentParser(description="el tespiti ile kisi sonuclarini karsilastirir")
p.add_argument("--rapor", default="capraz_islenmis.json",
               help="sonuclar/ altindaki capraz dogrulama raporu")
a = p.parse_args()
report_path = os.path.join(ROOT, "sonuclar", a.rapor)

if not os.path.exists(report_path):
    raise SystemExit("rapor yok, once cross_validation.py calistir")

with open(report_path) as f:
    report = json.load(f)

accuracy = np.array([r["test_dogrulugu"] for r in report["sonuclar"]])
videos = sorted(glob.glob(os.path.join(ROOT, "data", "videos", "*.mp4")))
signers = np.array([int(os.path.basename(v).split("_")[1]) for v in videos])
train = signers[signers <= 8]
val = signers[signers == 9]
test = signers[signers == 10]
signers = np.concatenate([train, val, test])

data_dir = os.path.join(ROOT, "data", report["veri"])
X = np.concatenate([np.load(os.path.join(data_dir, f"X_{name}.npy"))
                    for name in ["train", "val", "test"]])

detection = []
for s in range(1, 11):
    signer_data = X[signers == s]
    detection.append(((signer_data[:, :, 703] + signer_data[:, :, 704]) > 0).mean() * 100)
detection = np.array(detection)

r = np.corrcoef(detection, accuracy)[0, 1]

print("kisi  tespit  dogruluk")
for i in range(10):
    print(f"  {i+1:2d}   %{detection[i]:.1f}   %{accuracy[i]:.1f}")
print(f"\nkorelasyon: {r:.2f}")

plt.figure(figsize=(8, 6))
plt.scatter(detection, accuracy, s=120)
for i in range(10):
    plt.annotate(f"K{i+1}", (detection[i], accuracy[i]),
                 xytext=(8, 4), textcoords="offset points")

z = np.polyfit(detection, accuracy, 1)
xs = np.linspace(detection.min(), detection.max(), 50)
plt.plot(xs, np.poly1d(z)(xs), "r--", label=f"r={r:.2f}")

plt.xlabel("el tespit oranı (%)")
plt.ylabel("test doğruluğu (%)")
plt.grid(alpha=0.3)
plt.legend()
os.makedirs(os.path.join(ROOT, "sonuclar"), exist_ok=True)
plt.savefig(os.path.join(ROOT, "sonuclar", "tespit_dogruluk.png"), dpi=120)
plt.show()
