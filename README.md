# Can You Spot Fake Reviews?
Author: Mikhaela Martin


## Zaful Reviews Analysis
While I was working part-time as a high schooler, I spent a lot of money on clothing. Most of my attire was from online clothing stores such as SheIn and YesStyle. Although it was cheap, a lot of the pieces did not end up looking like what was portrayed online. The reviews were good but it did not match reality. Although the pictures and the comments looked legitatmate, I was upset.

I started this project for myself and others who like to shop online. While the reviews may seem positive, I knew they were not. I wanted to see whether i could analyze them and spot any patterns and anomalies. Here are my results.

### Project Objectives:
1. Identify patterns on reviews that could insinuate suspicious reviewer behavior

### Dataset
I webscraped Zaful Reviews from the [Floral Dresses](https://www.zaful.com/s/floral-dresses/) section of the website. These reviews were webscraped on ~September 18, 2020.

### Methods
- Cleaning Data: Random Imputation, Normalization, CountVectorizor, TFIDFVectorizor
- Modeling: Kmeans, PCA

*Clusters*

Individual Rating
Date (ordinal)
Color
Time (numerical)
Rank
Size
missing Waist
missing Height
dress
material

Cluster 1
material
fantastic
fantastic material
comfortable
comfortable fantastic
comfortable fantastic material
like
looks
pictures
like pictures

Cluster 2
super
cute
super cute
like
Individual Rating
pictures
like pictures
looks
exaclty like
exaclty

Cluster 3
material
size
comfortable
fits
true size
true
fantastic
fantastic material
comfortable fantastic material
comfortable fantastic

Cluster 4
like
exaclty
exaclty like
looks
pictures
like pictures
exaclty like pictures
Individual Rating
looks exaclty
looks exaclty like

Cluster 5
size
true
fits
true size
like
Individual Rating
looks
pictures
like pictures
size fits

Cluster 6
size
love
true
true size
like
pictures
cute
like pictures
exaclty like pictures
exaclty
