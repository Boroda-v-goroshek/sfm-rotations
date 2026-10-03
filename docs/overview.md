Image Matching Challenge 2025
Reconstruct 3D scenes from messy image collections.

Image Matching Challenge 2025

Late Submission
Overview
You'll develop machine learning algorithms that can figure out which images belong together and use them to reconstruct accurate 3D scenes. This innovation advances the field of computer vision and enables new applications in augmented reality, robotics, and AI.

Start

Apr 2, 2025
Close

Jun 3, 2025
Merger & Entry
Description
Note: This is the third Image Matching Challenge. It builds on 2024's Image Matching Challenge. This year, we’re taking it a step further and challenging you to determine how images should be grouped together or discarded, in addition to reconstructing 3D scenes.

Imagine sitting down to solve a jigsaw puzzle. But when you open the box, you discover that your pieces have been jumbled together with more pieces from other puzzle sets! How do you determine which pieces are for your puzzle and which belong to other sets?

Reconstructing a 3D scene from a set of possibly related images is a core problem in computer vision. While current methods work well in controlled environments with professional equipment, they struggle with real-world image collections.

Online image collections are messy and often contain a mix of unrelated photos or visually similar images that confuse reconstruction models. For example, two nearly identical sides of a monument might be mistaken for the same view, or completely unrelated images (like a photo of a latte taken near a landmark) could accidentally be grouped together. Existing methods use GPS data or video sequences to help, but these aren't always available or reliable, making large-scale applications unreliable.

This competition challenges you to identify which images should be grouped and which should be discarded in 3D scene reconstruction. This will improve Structure from Motion (SfM) techniques and help generate more accurate 3D models from diverse image collections.

Your work could make crowdsourced images more useful for large-scale reconstructions, benefiting areas like urban planning and scientific research.

Evaluation
The ground-truth data consists of multiple datasets
. Each contains one or more scenes
, with images
. The different scenes belonging to one dataset do not overlap: they show different regions or objects, but are similar in appearance, like two sides of the same building, or two different trees. Each dataset may have a different number of scenes and images. Each dataset may or may not contain "outlier" images, which do not correspond to any scene.

For training data, we provide the camera pose of each image in terms of its rotation matrix
and translation vector
. Images that do not correspond to any scene are placed in an outliers folder, for which no pose is provided. For test data, we combine all images into a single folder.

The task given to participants is (1) to partition the images
of each dataset
into a set of clusters
or outliers, thus restoring the original scene assignment; and (2) to reconstruct each scene independently, providing the camera pose of each image, expressed as a rotation matrix and a translation vector.

The submission score is obtained as the combination of the mean Average Accuracy (mAA) of the registered camera centers, and the clustering score. The mAA of a given cluster
with respect to the scene
is computed exactly, using the same metric as in the 2024 competition. It is equal to the ratio of the registered images of the scene found in the cluster divided by the cardinality of the scene. The clustering score is given by number images of the cluster effectively belonging to the scene, divided by the cardinality of the cluster, i.e.
.

The score computation involves two steps. In the first step, each scene
is greedily associated to the cluster
, among those provided by the user, that maximizes the mean Average Accuracy (mAA). If other user clusters provide the same mAA, the one with higher clustering score is chosen. The cluster index associated to scene
in this way is denoted by
. Notice that the same user cluster can be associated to more scenes, and that all the images labeled by the user as outliers are excluded from the greedy assignment.

In the second step, the overall mAA score and clustering score of a dataset are computed by aggregating the individual values. For the clustering score this is equal to

And analogously for the mAA score.

The combined score
is the harmonic mean of the mAA and clustering scores. In this formulation, the mAA score is roughly equivalent to recall, and the clustering score to precision, and the final score is thus analogous to the standard F1 score. Finally, we average the results over the different datasets to obtain a single score.

Submission file
For each image ID in the test set, you must predict a scene assignment and a pose. The file should contain a header and have the following format:

dataset,scene,image,rotation_matrix,translation_vector
dataset1,cluster1,image1.png,0.1;0.2;0.3;0.4;0.5;0.6;0.7;0.8;0.9,0.1;0.2;0.3
dataset1,cluster1,image2.png,0.1;0.2;0.3;0.4;0.5;0.6;0.7;0.8;0.9,0.1;0.2;0.3
dataset1,cluster2,image3.png,0.1;0.2;0.3;0.4;0.5;0.6;0.7;0.8;0.9,0.1;0.2;0.3
dataset1,cluster2,image4.png,0.1;0.2;0.3;0.4;0.5;0.6;0.7;0.8;0.9,0.1;0.2;0.3
dataset1,outliers,image5.png,nan;nan;nan;nan;nan;nan;nan;nan;nan,nan;nan;nan
dataset2,cluster1,image1.png,0.1;0.2;0.3;0.4;0.5;0.6;0.7;0.8;0.9,0.1;0.2;0.3
…
The scene labels are assigned by you. They are only used to specify which images belong together, so they can contain arbitrary strings: we recommend using something simple, like cluster1, cluster2, etc. The exception is an outliers label, which should be assigned to images that cannot be registered to any other image.

The example above contains dataset1 with five images. The solution indicates that images 1 and 2 belong together, as do images 3 and 4. Image 5 does not, so it is assigned to outliers. Images that you think are outliers are not registered to other images, so the rotation matrix and translation vector are not used: we make this clear using nan for every value.

The rotation_matrix (a 3x3 matrix) and translation_vector (a 3-D vector) are written as ;-separated vectors. Matrices are flattened into vectors in row-major order. Note that this metric does not require camera intrinsics, i.e., the calibration matrix
) that is usually estimated along with
and
during the 3D reconstruction process.

Note that you can group images together if you think they belong to the same scene, even if you cannot register them. You can indicate this by using the right scene label and nan values for rotation_matrix and translation_vector. For example:

dataset2,cluster1,image2.png,nan;nan;nan;nan;nan;nan;nan;nan;nan,nan;nan;nan
Notice that we can re-use scene labels (such as cluster1) for different datasets.

Timeline
April 1, 2025 - Start Date.

May 26, 2025 - Entry Deadline. You must accept the competition rules before this date in order to compete.

May 26, 2025 - Team Merger Deadline. This is the last day participants may join or merge teams.

June 2, 2025 - Final Submission Deadline.

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

Code Requirements

This is a Code Competition
Submissions to this competition must be made through Notebooks. In order for the "Submit" button to be active after a commit, the following conditions must be met:

CPU Notebook \<= 9 hours run-time
GPU Notebook \<= 9 hours run-time
Internet access disabled
Freely & publicly available external data is allowed, including pre-trained models
Submission file must be named submission.csv
Please see the Code Competition FAQ for more information on how to submit. And review the code debugging doc if you are encountering submission errors.

Prizes
1st Place - $12,000
2nd Place - $10,000
3rd Place - $10,000
4th Place - $10,000
5th Place - $8,000
CVPR 2025 Workshop
This competition is part of the Image Matching: Local Features and Beyond workshop at CVPR'25. Selected submissions to the competition will be invited to give talks at the workshop on June 11, 2025, in Nashville TN, USA (in person or remotely). Attending the workshop is not required to participate in the competition.

CVPR 2025 will be a hybrid conference. Attendees presenting in person are responsible for all costs associated with expenses and fees to attend CVPR 2025.

Citation
Fabio Bellavia, Jiri Matas, Dmytro Mishkin, Luca Morelli, Fabio Remondino, Amy Tabb, Eduard Trulls, Kwang Moo Yi, Sohier Dane, Addison Howard, and María Cruz. Image Matching Challenge 2025. https://www.kaggle.com/competitions/image-matching-challenge-2025, 2025. Kaggle.

Dataset Description
Building a 3-D model of a scene given an unstructured collection of images taken around it is a longstanding problem in computer vision research. Your challenge in this competition is to generate 3-D reconstructions from image sets showing different types of scenes and accurately pose those images. This year, we’re challenging you to determine how images should be grouped together or discarded, in addition to reconstructing 3D scenes.

This competition uses a hidden test. When your submitted notebook is scored, the actual test data (including a sample submission) will be made available to your notebook. Expect to find roughly 1,300 images in the hidden test set.

train_labels.csv A list of images in the training data, including ground truth.

dataset: The unique identifier for the dataset.
scene: The unique identifier for the scene.
image: The image filename.
rotation_matrix: The first target column. A 3x3 matrix, flattened into a vector in row-major convention, with values separated by ";". Sample values are random.
translation_vector: The second target column. A 3-D dimensional vector, with values separated by ";". Sample values are random.
train_thresholds.csv A list of thresholds for evaluation.

dataset: The unique identifier for the dataset.
scene: The unique identifier for the scene.
thresholds: A list of thresholds (in meters) for evaluation, separated by ";".
[train/test]/[dataset]/*.png A set of images taken from one or more locations. For instance, train/imc2024_lizard_pond contains images from two different scenes, lizard and pond. A folder may also contain single images that do not correspond to any scene ("outliers"). Note: We noticed that imc2023_haiper contains images from two scenes with spatial overlap: bike_image\_*.png and chairs_image\_\*.png. This was not intentional. You can consider this a hard case, or ignore it. [train/test]/[dataset]/LICENSE.txt The license for each dataset.

Note: The published test folder contains a subset of scenes from the training set. It is provided solely for example purposes. The folder will be populated with different data when submitting a notebook for scoring.

sample_submission.csv A valid, randomly-generated sample submission with the following fields:

image_id: A unique identifier for the row. Must be present in your submission file.
dataset: The unique identifier for the dataset.
scene: The unique identifier for the scene. In the submission file, this label is arbitrary and should be ignored. In your submission, it will indicate which images belong together (for a given dataset).
image: The image filename.
rotation_matrix: Same as above.
translation_vector: Same as above.
