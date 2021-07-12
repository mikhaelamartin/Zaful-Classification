# Can You Spot Fake Reviews?
Author: Mikhaela Martin


# Zaful Reviews Analysis

## Why Did I Start This Project?
While I was working part-time as a high schooler, I spent a lot of money on clothing. Most of my attire was from online clothing stores such as SheIn and YesStyle. Although it was cheap, a lot of the pieces did not end up looking like what was portrayed online. The reviews were good but it did not match reality. Although the pictures and the comments looked legitatmate, I was upset.

I started this project for myself and others who like to shop online. While the reviews may seem positive, I knew they were not. I wanted to see whether i could analyze them and spot any patterns and anomalies. Here are my results.

### Business Problem and Model Objective:
1. Identify patterns based on cluster segmentation

### Conclusions

Based on the clustering, I found no good insight being taken. The factors that ultimately made the most difference in the two clsuters were the missingness of data. Cluster 0 contained most of the samples that were missing data while Cluster 1 did not. Aside from that, reviews that were in cluster 0 had a slightly lower average date, meaning they were posted a little later than the other cluster. In terms of the text themselves, they both had very similar common words and phrases. They also shared a lot of duplicate comments.

Overall, the reviews from Zaful were tailored in a way that made my unsupervised model hard to detect patterns that could differentiate "real" and "fake" reviews.

Findings: After separating the reviews into two clusters, I found that the features that differentiated them the most were somewhat expected. In terms of the actual comment posted, there didn't seem to be much of a difference between the most common and relevant words found. For both clusters, some of the most common phrases were "material", "cute", "like", and "fits". But, the first cluster specifically also had the words "exaclty like pictures" (exactly is spelled wrong) so maybe these reviews could be spam.

![](https://github.com/mikhaelamartin/Zaful-Classification/blob/master/jupyter%20notebooks/All%20Features%20-%20Cluster%20WordClouds.png "All Features - Cluster Wordclouds")

In terms of the attributes of the reviews posted, there were major differences in these features: "Missing ". Because of this, I concluded that the date played a role in which features were missing. The earlier the date, the less amount of missing featurs.

## Clusters Using All Columns
I grouped the data into two clusters, as per the elbow method:

![](https://github.com/mikhaelamartin/Zaful-Classification/blob/master/jupyter%20notebooks/All%20Features%20-%20Elbow%20Plot.png "All Features - PCA Plot")

Most relevant words, according to TFIDFVectorizor:

![](https://github.com/mikhaelamartin/Zaful-Classification/blob/master/jupyter%20notebooks/All%20Features%20-%20Cluster%20WordClouds.png "All Features - Cluster Wordclouds")

Visualizing Clusters with PCA:

![](https://github.com/mikhaelamartin/Zaful-Classification/blob/master/jupyter%20notebooks/All%20Features%20-%20PCA.png "All Features - PCA")

Non-Text Feature Differences:
![](https://github.com/mikhaelamartin/Zaful-Classification/blob/master/jupyter%20notebooks/All%20Features%20-%20Non-Text%20Features%20Plot.png "All Features - Non-Text Feature Differences")


### Dataset
I webscraped Zaful Reviews from the [Floral Dresses](https://www.zaful.com/s/floral-dresses/) section of the website. These reviews were webscraped on ~September 18, 2020.

### Methods
- Cleaning Data: Random Imputation, Normalization, CountVectorizor, TFIDFVectorizor
- Modeling: Kmeans, PCA