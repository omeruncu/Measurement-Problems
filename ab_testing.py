# Basic Statistics Concepts
##### Without a grounding in Statistics, a Data Scientist is a Data Lab Assistant
##### The Future of AI will be about less data, Not More
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# Sampling

population = np.random.randint(0, 80, 10000)
population.mean()

np.random.seed(115)

sample = np.random.choice(a=population, size=100)
sample.mean()


np.random.seed(10)
sample1 = np.random.choice(a=population, size=100)
sample2 = np.random.choice(a=population, size=100)
sample3 = np.random.choice(a=population, size=100)
sample4 = np.random.choice(a=population, size=100)
sample5 = np.random.choice(a=population, size=100)
sample6 = np.random.choice(a=population, size=100)
sample7 = np.random.choice(a=population, size=100)
sample8 = np.random.choice(a=population, size=100)
sample9 = np.random.choice(a=population, size=100)
sample10 = np.random.choice(a=population, size=100)

(sample1.mean() + sample2.mean() + sample3.mean() + sample4.mean() + sample5.mean() +
 sample6.mean() + sample7.mean() + sample8.mean() + sample9.mean() + sample10.mean()) / 10

## Result: As the sample size increases, the mean of the sample distribution converges to the population mean.


# Descriptive Statistics

df = sns.load_dataset("tips")
df.describe().T

# Confidence Intervals
#### It is the finding of a range of two numbers that can cover the estimated value of the population parameter (statistic).

## Sample Mean
### Step 1: Find n, mean and standard deviation
### n = 100, mean = 180, standard deviation = 40
### Step 2: Decide on confidence interval: 95 or 99?
### Calculate Z table value (1,96 - 2,57)
### Step 3: Calculate Confidence interval using above values:
### x ± (z * s / √n) = 180 ± (1,96 * 40 / √100)
### result : 180 ± 7,84 so, between 172 and 188

## Tips Confidence Interval Calculation for Numerical Variables in Data Set

df.head()

# The restaurant owner can observe the average bill that customers will pay as a statistical range as a result of the following output:
sms.DescrStatsW(df["total_bill"]).tconfint_mean()

sms.DescrStatsW(df["tip"]).tconfint_mean()

## Confidence Interval Calculation for Numerical Variables in Titanic Dataset

df = sns.load_dataset("titanic")
df.describe().T

sms.DescrStatsW(df["age"].dropna()).tconfint_mean()

sms.DescrStatsW(df["fare"].dropna()).tconfint_mean()

# Correlation
# It is a statistical method that provides information about the relationship between variables, the direction and intensity of this relationship.

## Tip dataset:
## total_bill: total price of meal (including tip and tax)
## type: tip
## sex: gender of person paying (0=male, 1=female)
## smoker: is there anyone smoking in the group? (0=No, 1=Yes)
## day: day (3=Thur, 4=Fri, 5=Sat, 6=Sun)
## time: when? (0=Day, 1=Night)
## size: how many people in the group?

df = sns.load_dataset('tips')
df.head()

df["total_bill"] = df["total_bill"] - df["tip"]

df.plot.scatter("tip", "total_bill")
plt.show()

df["tip"].corr(df["total_bill"])

# Hypothesis Testing
# Statistical methods used to test an argument
# Our focus is on group comparisons, where the main goal is to try to show whether possible differences have arisen by chance.
# Example: Has the average daily time spent by users on the application increased after the interface change in the mobile application?
# A : users before the app was updated
# B : users after the app was updated
# h0: There is no difference in the daily average time spent on the application by users a and b
# h1: There is difference in the daily average time spent on the application by users a and b



# AB Testing (Independent Two Sample T Test)
# Used when comparison is desired between two group means.

## 1. Establish Hypotheses
## 2. Assumption Check
### 1. Normality Assumption
### 2. Variance Homogeneity
## 3. Applying Hypothesis
### 1. If assumptions are met, independent two-sample t-test (parametric test)
### 2. If assumptions are not met, Mannwhitneyu test (non-parametric test)
## 4. Interpret results according to p-value
## Note:
## - If normality is not met, number 2 directly. If variance homogeneity is not met, argument is entered for number 1.
## - It may be useful to perform outlier value examination and correction before normality examination.


## Application 1: Is There Any Difference Between the Average Calculations of Smokers and Non-Smokers?

df.groupby('smoker').agg({'total_bill': 'mean'})

### 1. Establish Hypotheses
#### H0: M1 = M2
#### H1: M1 != M2

### 2. Assumption Check
#### 1. Normality Assumption
##### H0: Normal distribution assumption is met.
##### H1:..not met.

test_stat, pvalue = shapiro(df.loc[df["smoker"] == "Yes", "total_bill"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

##### If p-value < 0.05 then HO REJECT.
##### If p-value > 0.05 then H0 CANNOT BE REJECTED.

test_stat, pvalue = shapiro(df.loc[df["smoker"] == "No", "total_bill"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

#### 2. Variance Homogeneity
##### H0: Variances are Homogeneous
##### H1: Variances are Not Homogeneous

test_stat, pvalue = levene(df.loc[df["smoker"] == "Yes", "total_bill"],
                           df.loc[df["smoker"] == "No", "total_bill"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

##### If p-value < 0.05 then HO REJECT.
##### If p-value > 0.05 then H0 CANNOT BE REJECTED.

### 3. Applying Hypothesis & 4. Interpret results according to p-value
#### 1. If assumptions are met, independent two-sample t-test (parametric test)

test_stat, pvalue = ttest_ind(df.loc[df["smoker"] == "Yes", "total_bill"],
                              df.loc[df["smoker"] == "No", "total_bill"],
                              equal_var=True)

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

##### If p-value < 0.05 then HO REJECT.
##### If p-value > 0.05 then H0 CANNOT BE REJECTED.

#### 2. If assumptions are not met, Mannwhitneyu test (non-parametric test)

test_stat, pvalue = mannwhitneyu(df.loc[df["smoker"] == "Yes", "total_bill"],
                                 df.loc[df["smoker"] == "No", "total_bill"])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

## Application 2: Is there a statistically significant difference between the average ages of Titanic's male and female passengers?

df = sns.load_dataset("titanic")
df.head()

df.groupby("sex").agg({"age": "mean"})

### 1. Set up hypotheses:
#### H0: M1 = M2 (There is no statistically significant difference between the average ages of male and female passengers)
#### H1: M1! = M2 (There is statistically...)

### 2. Assumption Check
#### 1. Normality Assumption
##### H0: Normal distribution assumption is met.
##### H1:..not met

test_stat, pvalue = shapiro(df.loc[df["sex"] == "female", "age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

test_stat, pvalue = shapiro(df.loc[df["sex"] == "male", "age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

#### 2. Variance Homogeneity
##### H0: Variances are Homogeneous
##### H1: Variances are Not Homogeneous

test_stat, pvalue = levene(df.loc[df["sex"] == "female", "age"].dropna(),
                           df.loc[df["sex"] == "male", "age"].dropna())

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

#### Nonparametric as assumptions are not met

test_stat, pvalue = mannwhitneyu(df.loc[df["sex"] == "female", "age"].dropna(),
                                 df.loc[df["sex"] == "male", "age"].dropna())

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

## Application 3: Is There a Difference in the Average Ages of Diabetic and Non-Diabetic Patients?

df = pd.read_csv("datasets/diabetes.csv")
df.head()

df.groupby("Outcome").agg({"Age": "mean"})

### 1. Set up hypotheses
#### H0: M1 = M2 -> There is no difference between the mean ages of diabetic and non-diabetic patients
#### H1: M1 != M2 -> There is a difference...

### 2. Assumption Check
#### 1. Normality Assumption (H0: Normal distribution assumption is provided.)
test_stat, pvalue = shapiro(df.loc[df["Outcome"] == 1, "Age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

test_stat, pvalue = shapiro(df.loc[df["Outcome"] == 0, "Age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

### Nonparametric since normality assumption is not met.

### Hypothesis (H0: M1 = M2)
test_stat, pvalue = mannwhitneyu(df.loc[df["Outcome"] == 1, "Age"].dropna(),
df.loc[df["Outcome"] == 0, "Age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))


## Business Problem: Are the Scores of Those Who Watched the Majority of the Course Different from Those Who Did Not Watch?

# H0: M1 = M2 (... there is no statistically significant difference between the two group means.)
# H1: M1 != M2 (... there is)

df = pd.read_csv("datasets/course_reviews.csv")
df.head()
df.describe().T

df[(df["Progress"] > 75)]["Rating"].mean()

df[(df["Progress"] < 25)]["Rating"].mean()


test_stat, pvalue = shapiro(df[(df["Progress"] > 75)]["Rating"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))


test_stat, pvalue = shapiro(df[(df["Progress"] < 25)]["Rating"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

test_stat, pvalue = mannwhitneyu(df[(df["Progress"] > 75)]["Rating"],
                                 df[(df["Progress"] < 25)]["Rating"])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))


# AB Testing (Two Sample Ratio Test)
# H0: p1 = p2
# H1: p1 != p2

# Application: Mobile app registration screen update
# Let's say 1100 people entered the old design screen and 250 people registered; 1000 people entered the new design screen and 300 people registered.

# H0: There is no statistically significant difference between the Conversion rate of the new design and the conversion rate of the old design
# H1: There is statistically ...
success_number = np.array([300, 250])
observation_numbers = np.array([1000, 1100])

proportions_ztest(count=success_number, nobs=observation_numbers)

success_number / observation_numbers

## Application: Is There a Statistically Significant Difference Between Survival Rates of Men and Women?

# H0: p1 = p2 -> There is no a statistically significant difference between the survival rates of women and men
# H1: p1 != p2 -> There is a statistically significant difference between the survival rates of women and men

df = sns.load_dataset("titanic")
df.head()

df.loc[df["sex"] == "female", "survived"].mean()

df.loc[df["sex"] == "male", "survived"].mean()

female_succ_count = df.loc[df["sex"] == "female", "survived"].sum()
male_succ_count = df.loc[df["sex"] == "male", "survived"].sum()

test_stat, pvalue = proportions_ztest(count=[female_succ_count, male_succ_count],
                                      nobs=[df.loc[df["sex"] == "female", "survived"].shape[0],
                                            df.loc[df["sex"] == "male", "survived"].shape[0]])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# H0 rejected

# ANOVA (Analysis of Variance)

## Used to compare more than two group means.
## Total bill difference test on a daily basis
df = sns.load_dataset("tips")
df.head()

df.groupby("day")["total_bill"].mean()

## 1. Set up hypotheses

### HO: m1 = m2 = m3 = m4 -> There is no difference between group means.
### H1: They are not equal at least one is different

## 2. Assumption check

### Normality assumption
### Variance homogeneity assumption

#### If assumption is met, one way anova
#### If assumption is not met, kruskal

#### H0: Normal distribution assumption is met.

for group in list(df["day"].unique()):
    pvalue = shapiro(df.loc[df["day"] == group, "total_bill"])[1]
    print(group, 'p-value: %.4f' % pvalue)

#### H0: Variance homogeneity assumption is met.

test_stat, pvalue = levene(df.loc[df["day"] == "Sun", "total_bill"],
                           df.loc[df["day"] == "Sat", "total_bill"],
                           df.loc[df["day"] == "Thur", "total_bill"],
                           df.loc[df["day"] == "Fri", "total_bill"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# 3. Hypothesis testing and p-value interpretation

# None of them are valid.
df.groupby("day").agg({"total_bill": ["mean", "median"]})


# HO: There is no significant difference between group averages

# parametric anova test:
f_oneway(df.loc[df["day"] == "Thur", "total_bill"],
         df.loc[df["day"] == "Fri", "total_bill"],
         df.loc[df["day"] == "Sat", "total_bill"],
         df.loc[df["day"] == "Sun", "total_bill"])

# Nonparametric anova test:
kruskal(df.loc[df["day"] == "Thur", "total_bill"],
        df.loc[df["day"] == "Fri", "total_bill"],
        df.loc[df["day"] == "Sat", "total_bill"],
        df.loc[df["day"] == "Sun", "total_bill"])


# difference detected and cause investigation
from statsmodels.stats.multicomp import MultiComparison

comparison = MultiComparison(df['total_bill'], df['day'])
tukey = comparison.tukeyhsd(0.05)
print(tukey.summary())

tukey = comparison.tukeyhsd(0.01)
print(tukey.summary())

tukey = comparison.tukeyhsd(0.10)
print(tukey.summary())