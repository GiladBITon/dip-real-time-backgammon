import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


cm = np.array([
    [144,12,2,0,2,4],
    [4,114,0,0,0,0],
    [0,4,154,8,2,2],
    [0,2,12,180,0,4],
    [0,4,4,19,168,8],
    [0,0,4,9,0,158],
])

cm_fixed = np.array([
    [130,2,0,0,0,0],
    [0,130,0,0,0,0],
    [0,0,160,0,0,0],
    [0,0,0,193,0,0],
    [0,0,0,3,140,0],
    [0,0,0,2,0,165],
])

plt.figure(figsize=(10,8))
plt.subplot(2,2,1)
sns.heatmap(cm_fixed, annot=True, fmt="d", cmap="Blues", xticklabels=["1","2","3","4","5","6"], yticklabels=["1","2","3","4","5","6"])
plt.xlabel("True Dice Number"), plt.ylabel("Detected Dice Number"), plt.title("Dice Number Detection Confusion Matrix")
plt.subplot(2,2,2)
sns.heatmap(cm_fixed / 918, annot=True, fmt=".3f", cmap="Blues", xticklabels=["1","2","3","4","5","6"], yticklabels=["1","2","3","4","5","6"])
plt.xlabel("True Dice Number"), plt.ylabel("Detected Dice Number"), plt.title("Normalized Dice Number Detection Confusion Matrix")
plt.tight_layout()
plt.savefig('/Users/razbarak/Downloads/cm_fixed.png', dpi=300, bbox_inches='tight')
plt.show()