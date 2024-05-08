# Asymmetry Feature

Asymmetry feature.

Current standards define asymmetry in terms of color, shape, and others [CITE PAPER you read) but in our paper we focus on shape asymmetry only, because of time constraints. In addition, we take into account specifically on line symmetry, not on rotational symmetry (Picture showing the difference), because it is the type of symmetry used in ABCD checklist which we treat this as a baseline. 

We approached the problem by first proposing three methods ourselves based on pure mathematical properties of symmetry and then by performing research on similar papers to arrive 

Assessing line symmetry boils down to two steps:

1. Finding axis which could be called an axis of symmetry
2. folding the picture alongside that axis and computing the overlap.

Actually the biggest challenge is, how to define axis of symmetry and this is the main way these methods differ. Below we present detailed description of algorithms. At the end we performed the test to choose one best method - actually we ended up with the forth method.

| Codename | Description | Disadvantages |
| --- | --- | --- |
| Fully rotated aggregated symmetry | Rotates an image 180 times and computes average overlap proportion for each of the rotatations. Axes go through middle of the image. | Elippsis resembling shapes with very different major and minor axis will get low score even though are highly symmetrical. |
| Max Symmetry | Rotates an image 180 and returns overlap proportion for the axis which yields greatest overlap. Axis goes through the middle of the image. | It finds only one line of symmetry. Shapes with more than one line of symmetry won’t get higher score which intuitively should. |
| Major minor symmetry axis | Builds upon max symmetry. Finds ‘major’ axis as the axis alongside which folding yields highest overlap, but also does the fold alongside ‘minor axis’ defined as the axis perpendicalar to major one. Axes go through middle of the image. | Assumes that the second, minor axis of symmetry is perpendicular to the main. Might now work for shapes like 5-stars. |

Fourth way was obtained via research. Great review on measuring asymmetry we found in paper (cite) and we decided on Sirakov and their minimum bounding box method. That is, symmetry line is chosen to be the line crossing the image in parallel to the greater in value dimension (height or width) for the rotation of the image for which the bounding box has the smallest area.

<DRAW IO GRAPHICS SHOWING THE WAY ALGORITHM WORKS>

At the end we performed tests:

TESTS RESULTS

Which yielded that 4th method is the best and was used in our code.
