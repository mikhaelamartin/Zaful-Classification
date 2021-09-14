# Can You Spot Fake Reviews?
Author: Mikhaela Martin


# Zaful Reviews Analysis

### Project Goals:

**Business Problem**: Potential customers should be able to trust reviews on items they are interested in. But, they can have a hard time distinguishing real reviews and 'fake' reviews. Cheap clothing websites are notorious for spamming their site with fake reviews, so I will be conducting my analysis on Zaful (www.zaful.com). 

**Model Objectives**: I will use clustering methods to detect patterns, spot anomalies, and visualize results on Zaful reviews.

### Dataset
I webscraped Zaful Reviews from the [Floral Dresses](https://www.zaful.com/s/floral-dresses/) section of the website. These reviews were webscraped on ~September 18, 2020.

### Methods
- Cleaning Data: Random Imputation, Normalization, CountVectorizor, TFIDFVectorizor
- Modeling: Kmeans, PCA

### Conclusions

The clusters were not distinguishable enough to conclude strong results. The features that differentiated them the most was missing data. Cluster 0 mostly consisted of samples with missing data while Cluster 1 did not. Cluster 0 had a slightly lower average date posted, meaning they were posted slightly later than the other cluster. Therefore, earlier comments had less missing features. This might be because the review standard changed, the company automated unfilled attributes, or other unknown reasons. 

In terms of the text themselves, they both had very similar common words and phrases. They also shared a lot of duplicate comments. For both clusters, some of the most common phrases were "material", "cute", "like", and "fits". But, the first cluster specifically also had the words "exaclty like pictures" (exactly is spelled wrong) so these reviews could be spam.

#### The Optimal Number of Features is 2
![](https://github.com/mikhaelamartin/Zaful-Classification/blob/master/jupyter%20notebooks/All%20Features%20-%20Elbow%20Plot.png "All Features - PCA Plot")

#### Both Clusters Contained Similar Words
![](https://github.com/mikhaelamartin/Zaful-Classification/blob/master/jupyter%20notebooks/All%20Features%20-%20Cluster%20WordClouds.png "All Features - Cluster Wordclouds")

#### Despite Optimal Clustering, the Data Was Still Hard to Group
PC1 contains more than 26% of the variation in data, yet the two clusters looked like four. 
![](https://github.com/mikhaelamartin/Zaful-Classification/blob/master/jupyter%20notebooks/All%20Features%20-%20PCA.png "All Features - PCA")
P
PC1 

Non-Text Feature Differences:

#### Clusters Differ Most Because of Missing Values
Missing Overall Fit, Missing Height,and Missing Weight Differed the Most Between Clusters

![](https://github.com/mikhaelamartin/Zaful-Classification/blob/master/jupyter%20notebooks/All%20Features%20-%20Non-Text%20Features%20Plot.png "All Features - Non-Text Feature Differences")


