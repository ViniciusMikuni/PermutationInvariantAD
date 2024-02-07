import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

pQ_qcd = np.load("Pq_QCD.npz")["probs"]
pQ_top = np.load("Pq_Top.npz")["probs"]
pT_qcd = np.load("Pt_QCD.npz")["probs"]
pT_top = np.load("Pt_Top.npz")["probs"]

plt.hist(pQ_qcd, histtype="step", bins=100, density=True, color="r", label=r"$p_{QCD}(QCD)$")
plt.hist(pQ_top, histtype="step", bins=100, density=True, color="b", label=r"$p_{QCD}(Top)$")
plt.hist(pT_qcd, histtype="step", bins=100, density=True, color="r", linestyle="--", label=r"$p_{Top}(QCD)$")
plt.hist(pT_top, histtype="step", bins=100, density=True, color="b", linestyle="--", label=r"$p_{Top}(Top)$")
plt.legend(loc="upper left")
plt.ylim(0, 5e-3);

plt.figure()
lr_qcd = pT_qcd - pQ_qcd
lr_top = pT_top - pQ_top
plt.hist(lr_qcd, bins=np.linspace(-40, 25, 40), histtype="step", density=True, color="red", label="QCD")
plt.hist(lr_top, bins=np.linspace(-40, 25, 40), histtype="step", density=True, color="blue", label="Top")
plt.legend()
plt.xlim(-30, 30)


fpr, tpr, _ = roc_curve(np.append(np.zeros(len(lr_qcd)), np.ones(len(lr_top))), np.append(lr_qcd, lr_top));
plt.figure()
plt.plot(tpr, 1. / (fpr+1e-8),)
plt.yscale("log")
plt.ylim(0.9, 1e4)
plt.grid(which="both")
plt.xlim(0, 1);