import numpy as np

path = '/home/vboxuser/final-project/ROM_DATA/classic/snapshots/snapshot_0000.npz'
data = np.load(path)
freqs = data['freqs_hz']
vecs = data['eigvecs_real']

print(f"{'Freq (Hz)':<12} | {'Energy':<12} | {'Uniqueness':<12}")
print('-' * 45)

for i in range(1, len(freqs)):
    # נורמליזציה של הווקטורים להשוואת צורה
    v1 = vecs[:, i-1] / (np.linalg.norm(vecs[:, i-1]) + 1e-15)
    v2 = vecs[:, i] / (np.linalg.norm(vecs[:, i]) + 1e-15)
    
    # חישוב דמיון (מכפלה סקלרית) - ככל שקרוב ל-1 הם דומים יותר
    similarity = np.abs(np.dot(v1, v2))
    uniqueness = 1.0 - similarity
    energy = np.mean(np.abs(vecs[:, i]))
    
    # נדפיס את כל המודים הצפופים בהתחלה (עד 110Hz) ואז דגימות
    if freqs[i] < 110 or i % 20 == 0:
        print(f"{freqs[i]:12.2f} | {energy:12.2e} | {uniqueness:12.4f}")

# חישוב סטטיסטיקה כללית
avg_uniqueness = np.mean([1.0 - np.abs(np.dot(vecs[:, j-1]/np.linalg.norm(vecs[:, j-1]), vecs[:, j]/np.linalg.norm(vecs[:, j]))) for j in range(1, len(freqs))])
print(f"\nAverage Uniqueness: {avg_uniqueness:.4f}")
