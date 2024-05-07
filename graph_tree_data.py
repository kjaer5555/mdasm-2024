import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

file="best_tree_extract.csv"
df = pd.read_csv(file)

depths = np.array(df["depth"])
acc = np.array(df["accuracy"])
std = np.array(df["std_accuracy"])
prec = np.array(df["precision"])
recall = np.array(df["recall"])
f1_score = np.array(df["f1_score"])

sns.set_style("whitegrid")
plt.figure(figsize=(15,10))
plt.title('The optimal tree depth for 4500 trees', fontsize=20, fontweight='bold')
plt.xlabel('Depth of threes', fontsize=15)
plt.ylabel('Misclassification Error', fontsize=15)
plt.plot(depths, 1-acc,color="red")
plt.legend(["error"], loc="upper right")

plt.show()
