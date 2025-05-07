# Patterns of Regional Integration in Europe through Erasmus+ Student Mobility: the Effects of Covid-19

This is a repository for Python scripts used to analyse Erasmus+ student mobility data, the main data in my thesis.

## Pre-requisites

### Data
1. Spatially enriched Erasmus+ mobility data between 2014 and 2022 by [Väisänen et al.](https://doi.org/10.1038/s41597-025-04789-0)
2. Statistical NUTS 3 region data for the year 2021 by [GISCO](https://ec.europa.eu/eurostat/web/gisco/geodata/statistical-units/territorial-units-statistics), contains geographical typologies as well
3. Socio-economic data between 2014 and 2022 by [Eurostat](https://ec.europa.eu/eurostat/data/database). These include GDP, employment and population per NUTS 3 region

### Python libraries
1. pandas
2. skbio
3. sklearn
4. tslearn
5. math
6. numpy
7. matplotlib.pyplot
8. seaborn
9. Counter (from collections)
10. StandardScaler (from sklearn.preprocessing)
12. DBSCAN (from sklearn.cluster)
12. silhouette_samples (from sklearn.metrics)
13. TimeSeriesKMeans, silhouette_score (from tslearn.clustering)
14. cdist_dtw (from tslearn.metrics)

## Order of running the scripts

| Step   | File     | Description | Input |
| :----- | :------ | :--------- | :--- |
| 1      |   NUTS3.py  | Reads the statistical NUTS region data and cleans it     |    CSV file from GISCO  |
| 2      |   Regional_characteristics.py   | Reads the socio-economic data and cleans it       |   CSV files from Eurostat   |
| 3      |  Data analysis.py   | Performs all the analysis and visualization       |   CSV file of Erasmus+ student mobility (origin-destination matrix), cleaned dataframes from previous steps   |




## Link to author's LinkedIn
www.linkedin.com/in/maijahavusela


```python

```
