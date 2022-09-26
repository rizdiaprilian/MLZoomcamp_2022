### Week 3: Classification

This covers aspects of choosing a binary class according to the statistical distribution of features. Concepts and principles important to understand:

- Categorical variables to be encoded to binary representation resembling ON/OFF state for each unique value. This process is carried out with `DictVectorizer`.
- Mutual information tells how much categorical variables have their own importance contributed to the model based on their dependencies with the target variable. Meanwhile correlation (usually with Pearson or Spearman) analyses the magnitude of association amongst numerical variables. 
- Risk ratio to measure relative comparison between the risk that a single group have may and the risk among the entire population.
- Assessing performance in accuracy-wise standpoint leaves a large gap in explaining how many the model hits correct predictions and misses against the actual outcomes as defined in hypothesis (type 1 and type 2 error). That is where precision, recall, ROC-AUC emerge to describe this .