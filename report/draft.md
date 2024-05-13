# Dataset information

Our dataset is based on 978 photos from ……add citation…….. dataset. There are photos of all 6 diagnoses - BCC, ASK, SEK, MEL, NEV, SCC. We have used manually created masks ……possibly add automatic segmentation part…….. Due to the fact that the dataset sometimes contains more than one photo of the same patient, and sometimes even of the same lesion, we have made sure that all the photos of single patient are treated as a cohesive group. After partitioning the data into training and testing subsets, these groups remain intact within the same split. We have achieved that using sklearn.model_selection.StratifiedGroupKFold, which also maintains the proportion of the classes within the splits. In our case, that would mean the proportion of cancerous and non-cancerous lesions. At the end, the dataset is split in the following manner:
......Add photo with the table........


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


# Relative presence of important colors


During our research, we have found two different approaches for the color extraction - the one in the article by Kasmi and Mokrani (2016) [1] and the one in the article by Ali, Li, O'Shea (2020) [2]. In both articles, the authors have identified 6 suspicious colors, whose presence in a lesion is an indication of a cancerous disease. These are light-brown, dark-brown, red, white, black and blue-gray. In the first article, using the RGB color space, the authors have chosen one single value for each of these colors. After that, they calculate the Euclidian distance between each pixel of the lesion and all these color values. The smallest distance indicates that the pixel is closest to that color. Then they set a threshold of 5%, stating that if at least 5% of the pixels in the lesion are close to a certain color, then this color is present on the lesion. 
The second article has a similar approach, but they have used the CIELab color space as it is considered to more closely represent the human perception of the colors. They have used the Minkowski distance, which is a generalization of the Euclidian and Manhattan distances, and they have used an interval to define each color instead of a single value. We have instead used HSV color space, and because of that, we have to do a few modifications in order to match the methodology.
RGB is structured as a cube of size 255 and each of the three values, R, G and B, work as amount from each ingredient. Due to this nature, using Euclidian or Manhattan distance does not make much difference. HSV on the other hand, is structured as a cylinder of radius and height 100, and the three values work independently. The same change in each of them is comprehended differently by the human eye. For example, a change of 40 in the Saturation makes the color look either milder or sharper, but it remains essentially the same. That change in Hue however, might change green to yellow. Due to this independence, Manhattan distance is better, as it treats each of the components individually and incorporates their scales separately.
For similar reasons, choosing one reference value for each color in RGB color space and an interval for CIELab can work perfectly. In HSV, however, due to H being from 0 to 360 degrees, we face a problem around the borders, as the color [359,x,y] and the color [2,x,y] are almost indistinguishable for the human eye, but their Manhattan distance is 357, so they seem to be very different for the computer. 

Therefore, we have created 5 separate bins for the 6 suspicious colors/dark-brown and light-brown have been combined, as the low quality of many of the photos and the shadows make it very hard to define a border between the two colors/. After picking color shades from various lesions and then testing these shades for similarity in We have found 13 different reference colors - 3/3/3/2/2 for red/ pink/ brown/ white/ blue-gray, chosen after a process of color picking from different lesions, and then
In order to overcome this obstacle and to implement the ideas of the two articles, we have f
After a consensus, we chose a few lesions which have a large representation of each color. Then using a color picker, we have found the exact shades of these colors. After further testing, we chose 13 different shades (3/3/3/2/2 for red/ pink/ brown/ white/ blue-gray)

[1] Kasmi, R. and Mokrani, K. (2016), Classification of malignant melanoma and benign skin lesions: implementation of automatic ABCD rule. IET Image Processing, 10: 448-455. https://doi.org/10.1049/iet-ipr.2015.0385
[2] Ali AR, Li J, O'Shea SJ. Towards the automatic detection of skin lesion shape asymmetry, color variegation and diameter in dermoscopic images. PLoS One. 2020 Jun 16;15(6):e0234352. doi: 10.1371/journal.pone.0234352. PMID: 32544197; PMCID: PMC7297317.

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


