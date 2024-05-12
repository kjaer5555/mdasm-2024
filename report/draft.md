# Asymmetry Feature

Apparently asymmetry is a well-defined term, but actually it can carry several meanings.

Current protocols to assess lesion’s dangernousness very often include asymmetry of the lesion, though they define them differently. In ABCD rule color, shape and structure symmetry is taken into account[1] whereas in Menzies method [2] only symmetry of the pigmentation pattern is used. In addition, there are two different types of symmetry - rotational and line one, but in our research we have not been able to identify a paper which would focus on a rotational. For a review of main methods we recommend [3].

In our paper we focus on shape line asymmetry only, because it offered a good performance-time tradeoff. Stoecker et al. [4] developed method basing only on a shape line symmetry and it in 93.5% agreed with a dermatologist’s annotations.

Assessing line symmetry comprises two main steps:

1. Finding axis which could be called an axis of symmetry
2. folding the picture alongside that axis and computing the overlap.

We have approached the problem by first proposing three methods ourselves based on pure mathematical properties of line symmetry and then by trying out minimum bounding box method [5] The main challenge and difference between these methods is the first step. 

| Codename | Description | Disadvantages |
| --- | --- | --- |
| Fully rotated aggregated symmetry | Mask is rotated around center of its bounding box. It computes average overlap percentage for each of the rotation angle. Fold axes go through middle of the image. | Elippsis resembling shapes with very different major and minor axis will get low score even though are highly symmetrical. |
| Max Symmetry Axis | Mask is rotated around center of its bounding box. It returns overlap percentage for fold algongside axis which yields greatest overlap. Fold axis goes through the middle of the image. | It finds only one line of symmetry. Shapes with more than one line of symmetry won’t get higher score which intuitively should. |
| Max Major + perpendicular minor symmetry axes | Mask is rotated around center of its bounding box. Major axis is defined as the axis alongside which fold yields highest overlap (like above). Minor axis is perpendicular to the major axis. The function returns average overlap percentage for folds alongside both axes. Axes go through middle of the image. | Assumes that the second, minor axis of symmetry is perpendicular to the main. Might now work for shapes like 5-stars. |
| Minimum Bouding Box Method | Mask is rotated around center of its bounding box. Mask is so rotated that the bounding box has the smallest area. It returns overlap percentage for the fold alongside the axis which goes through center of the image and is parallel to the longer dimension (either width or height) | Also doesn’t capture if the object has more than one line of symmetry. |

*Caption: Fig n or sth*

At the end we performed tests to see how well do each of these methods can differentiate the data:

< RESULTS >

TESTS RESULTS

**I need here Sune to walk me through the code as I can barely understand what’s going on here.**

Which yielded that 4th method is the best and was used in our code.

[1] dermoscopedia contributors – Michael Kunz, Wilhelm Stolz, "ABCD rule," *dermoscopedia,* [https://dermoscopedia.org/w/index.php?title=ABCD_rule&oldid=20411](https://dermoscopedia.org/w/index.php?title=ABCD_rule&oldid=20411) (accessed May 12, 2024).

[2] dermoscopedia contributors – Scott Menzies, Ralph Braun, "Menzies Method," *dermoscopedia,* [https://dermoscopedia.org/w/index.php?title=Menzies_Method&oldid=9988](https://dermoscopedia.org/w/index.php?title=Menzies_Method&oldid=9988) (accessed May 12, 2024).

[3]  Talavera-Martínez, Lidia, Pedro Bibiloni, Aniza Giacaman, Rosa Taberner, Luis Javier Del Pozo Hernando, and Manuel González-Hidalgo. "A Novel Approach for Skin Lesion Symmetry Classification with a Deep Learning Model." *Computers in Biology and Medicine* 145 (2022): 105450. [https://doi.org/10.1016/j.compbiomed.2022.105450](https://doi.org/10.1016/j.compbiomed.2022.105450).

[4] Stoecker, William V., William Weiling Li, and Randy H. Moss. "Automatic Detection of Asymmetry in Skin Tumors." *Computerized Medical Imaging and Graphics* 16, no. 3 (1992): 191-197. [https://doi.org/10.1016/0895-6111(92)90073-I](https://doi.org/10.1016/0895-6111(92)90073-I).

[5] Sirakov, N. M., M. Mete, and N. S. Chakrader. "Automatic Boundary Detection and Symmetry Calculation in Dermoscopy Images of Skin Lesions." In *2011 18th IEEE International Conference on Image Processing*, 1605-1608. Brussels, Belgium: IEEE, 2011. https://doi.org/10.1109/ICIP.2011.6115757.

# Predicitve power of the model

The term ‘predictive power’ refers to the ability of a *model* (any type of model) to differentiate or predict true classes. Most common metric to quantify it are: AUC (Area under the curve) and Gini Coefficient, which can be treated as normalised AUC since $GINI=2\cdot AUC-1$, where $AUC \in [0.5, 1]$. It is often used in decision trees.

Given the binary classification problem - which is the case for our study - the model outputs probabilities of how much given sample belongs to the positive class. AUC score of the model is a probability that model assigns higher probability to a randomly selected positive class than to a randomly selected negative class. It has very useful property that it’s threshold invariant like other common metrics (accuracy, recall, etc.). It yields 1 of two classes are ideally separable i.e. in the two-features setting, with samples plotted, a line separating them perfectly could be put. 0.5 means model can differentiate, it’s random.

Our model yields AUC score of 0.76.

Other way to interpret AUC is as the area under the ROC curve. This curve plots Recall and False Positive Rate for all classification thresholds. Recall is true positives over all positives and FPR is false positives over all negatives. Each point on the curve represents a certain threshold and on axes are corresponding recall and FPR. Ideal state would be, if such a point was in the top-left corner - it would mean, the model both perfectly finds the all positive classes (recall=1) and doesn’t misclassify any negatives (FPR=1). Accuracy would be then 1 as well. 

We computed the threshold which is the closest to this top-left corner which is often referred as ‘optimal threshold’. It tries to both optimize recall and minimise FPR. We got 0.54

| Metric | Threshold=0.5 | Optimal threshold=0.53644 |
| --- | --- | --- |
| Accuracy | 0.725 | 0.738 |
| Recall | 0.785 | 0.764 |

Since the changes were insignificant and we favour recall over accuracy we didn’t choose this optimal threshold.

![Untitled](Raport%206305b3af738c4cbeb34dd713375495b1/Untitled.png)

![Untitled](Raport%206305b3af738c4cbeb34dd713375495b1/Untitled%201.png)

# Automatic Segmentation

### Logistic regression

