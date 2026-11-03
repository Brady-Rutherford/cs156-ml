#picture importing!
import glob, os, urllib.request
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from sklearn.decomposition import PCA

# ---------- 1. Download ("curl") the 10 phone photos from the web ----------
BASE = ("https://raw.githubusercontent.com/Brady-Rutherford/cs156-ml/"
        "claude/awesome-wright-nmezr1/PCW/PCA%20-%20Eigenshirts/phone_photos/")
os.makedirs("phone_photos", exist_ok=True)
for i in range(1, 11):
    name = f"photo{i:02d}.jpg"
    if not os.path.exists(f"phone_photos/{name}"):
        urllib.request.urlretrieve(BASE + name, f"phone_photos/{name}")   # same as: curl -o
paths = sorted(glob.glob("phone_photos/*.jpg"))                          # glob collects the files
print(len(paths), "photos:", paths)

# ---------- 2. Look at the raw arrays: every photo is a grid of pixels ----------
raw = np.asarray(Image.open(paths[0]).convert("RGB"))
print("\nraw photo 1 shape (height, width, RGB):", raw.shape, raw.dtype)
print("raw sizes differ:", [Image.open(p).size for p in paths[:4]], "...")

# ---------- 3. Pre-process: crop to a square and resize so all photos match ----------
SIZE = 64
def load(path):
    img = Image.open(path).convert("RGB")
    img = ImageOps.fit(img, (SIZE, SIZE))          # center crop to square, then resize
    return np.asarray(img, dtype=np.float32) / 255.0

images = np.array([load(p) for p in paths])        # shape (10, 64, 64, 3)
print("\nall photos stacked:", images.shape)

# ---------- 4. Splitting the arrays: pixels, channels, patches, rows ----------
img = images[0]
print("\none photo:", img.shape, "= 64 rows x 64 columns x 3 colors")
print("top-left pixel [R, G, B]:", img[0, 0])
print("centre pixel   [R, G, B]:", img[32, 32])
red, green, blue = img[:, :, 0], img[:, :, 1], img[:, :, 2]
print("one colour channel:", red.shape, " mean red / green / blue:",
      red.mean().round(3), green.mean().round(3), blue.mean().round(3))
patch = img[:8, :8]                                # an 8x8 corner patch
print("8x8 patch:", patch.shape)
gray = img.mean(axis=2)                            # average the 3 channels
print("grayscale version:", gray.shape)
row = img.reshape(-1)                              # flatten one photo into one long row
print("flattened photo:", row.shape, "= 64*64*3 numbers, first 6:", row[:6].round(3))

X = images.reshape(len(images), -1)                # one row per photo, ready for sklearn
print("data matrix X:", X.shape)

fig, axes = plt.subplots(1, 6, figsize=(14, 2.6))
axes[0].imshow(img); axes[0].set_title("64x64 photo")
for ax, ch, cmap, t in zip(axes[1:4], [red, green, blue], ["Reds", "Greens", "Blues"], ["red", "green", "blue"]):
    ax.imshow(ch, cmap=cmap, vmin=0, vmax=1); ax.set_title(f"{t} channel")
axes[4].imshow(gray, cmap="gray"); axes[4].set_title("grayscale")
axes[5].imshow(patch, interpolation="nearest"); axes[5].set_title("8x8 pixel patch")
for ax in axes: ax.axis("off")
plt.suptitle("Splitting one photo into pixels and channels")
plt.savefig("ext_1_splitting.png", dpi=150, bbox_inches="tight"); plt.show()

# ---------- 5. PCA on the 10 photos ----------
pca = PCA().fit(X)                                 # 10 photos -> at most 10 components
cum = np.cumsum(pca.explained_variance_ratio_)
print("\ncumulative explained variance:", cum.round(3))
k = int(np.argmax(cum >= 0.90) + 1)
print(f"components for 90% variance: k = {k} (out of {X.shape[1]} pixels)")

def to_img(v):
    v = v.reshape(SIZE, SIZE, 3)
    return (v - v.min()) / (v.max() - v.min())

fig, axes = plt.subplots(2, 5, figsize=(13, 5.5))
for i, ax in enumerate(axes.flat):
    ax.imshow(images[i]); ax.set_title(f"photo {i + 1}"); ax.axis("off")
plt.suptitle("The 10 photos after crop + resize")
plt.savefig("ext_2_photos.png", dpi=150, bbox_inches="tight"); plt.show()

fig, axes = plt.subplots(1, 6, figsize=(14, 2.8))
axes[0].imshow(pca.mean_.reshape(SIZE, SIZE, 3)); axes[0].set_title("mean")
for i, ax in enumerate(axes[1:]):
    ax.imshow(to_img(pca.components_[i]))
    ax.set_title(f"PC {i + 1} ({pca.explained_variance_ratio_[i]:.0%})")
for ax in axes: ax.axis("off")
plt.suptitle("Mean photo and first eigen-photos")
plt.savefig("ext_3_eigenphotos.png", dpi=150, bbox_inches="tight"); plt.show()

levels = [1, 3, k]
fig, axes = plt.subplots(len(levels) + 1, 5, figsize=(11, 2.3 * (len(levels) + 1)))
for col in range(5):
    axes[0, col].imshow(images[col])
    for r, n in enumerate(levels, start=1):
        p = PCA(n).fit(X)
        rec = np.clip(p.inverse_transform(p.transform(X)), 0, 1)
        axes[r, col].imshow(rec[col].reshape(SIZE, SIZE, 3))
axes[0, 0].set_ylabel("original")
for r, n in enumerate(levels, start=1): axes[r, 0].set_ylabel(f"{n} PCs")
for ax in axes.flat: ax.set_xticks([]); ax.set_yticks([])
plt.suptitle("Photos rebuilt from the reduced representation")
plt.savefig("ext_4_reconstructions.png", dpi=150, bbox_inches="tight"); plt.show()

Z = pca.transform(X)
plt.figure(figsize=(6, 5))
plt.scatter(Z[:, 0], Z[:, 1])
for i, (a, b) in enumerate(Z[:, :2]): plt.annotate(f"photo {i + 1}", (a, b))
plt.xlabel("PC 1"); plt.ylabel("PC 2"); plt.title("Photos in the first two PCs")
plt.savefig("ext_5_scatter.png", dpi=150, bbox_inches="tight"); plt.show()
