import pandas as pd
import warnings
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import minmax_scale
from sklearn.feature_selection import RFECV


def ignore_warn(*args, **kwargs):
    pass


warnings.warn = ignore_warn


def check_missing(df):
    missing_percentage = df.isnull().sum() / df.shape[0] * 100
    missing_order = missing_percentage.sort_values(ascending=False)
    missing_order.plot.bar()
    return missing_order


def check_unique(series):
    unique_values = series.value_counts(ascending=False)
    return unique_values


def pivot_table_plot(df, index, values):
    pivot_table = df.pivot_table(index=index, values=values)
    pivot_table.plot.bar()
    return pivot_table


titanic_train = pd.read_csv("E:/datasets/Titanic/train.csv")
titanic_test = pd.read_csv("E:/datasets/Titanic/test.csv")

ntitanic_train = titanic_train.shape[0]
titanic = pd.concat([titanic_train, titanic_test], axis=0, sort=False).reset_index(drop=True)
missing = check_missing(titanic)
# deal with missing
embarked = check_unique(titanic["Embarked"])
titanic["Embarked"] = titanic["Embarked"].fillna("S")
fare = check_unique(titanic["Fare"])
titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].mode()[0])

sex_pivot = pivot_table_plot(titanic, index="Sex", values="Survived")
pclass_pivot = pivot_table_plot(titanic, index="Pclass", values="Survived")
age_pivot = pivot_table_plot(titanic, index="Age", values="Survived")

# age categorizing
titanic["Age"] = titanic["Age"].fillna(-0.5)
titanic["Age_category"] = pd.cut(titanic["Age"], bins=[-1,0,5,12,18,35,60,100],
                                 labels=["Missing", "Infant", "Child", "Teenager", "Young Adult", "Adult", "Senior"])

titanic["Pclass"] = titanic["Pclass"].apply(str)
# predict with age_category; sex, pclass
titanic_dummies = pd.get_dummies(titanic[["Age_category", "Sex", "Pclass"]])

# train = titanic_dummies[:ntitanic_train]
# all_X1 = train
# all_y1 = titanic_train["Survived"]

holdout = titanic_dummies[ntitanic_train:]


def cross_validation(model, cv, whole_dataset):
    train = whole_dataset[:ntitanic_train]
    holdout = whole_dataset[ntitanic_train:]
    all_X = train
    all_y = titanic_train["Survived"]
    train_X, test_X, train_y, test_y = train_test_split(all_X, all_y, random_state=42, test_size=0.2)
    rmse = np.sqrt(cross_val_score(model, train_X, train_y, cv=cv))
    rmse_avg = np.mean(rmse)
    return rmse_avg


# predict with three dummies variable("age_category, sex, pclass")
lr = LogisticRegression()
score = cross_validation(lr, cv=10, whole_dataset=titanic_dummies)
print("the first time prediction {:.4f}".format(score))
# lr.fit(all_X, all_y)
# predictions = lr.predict(holdout)

# deal with sibsp, parch, embark and fare
titanic[["SibSp_scaled", "Parch_scaled", "Fare_scaled"]] = \
    titanic[["SibSp", "Parch", "Fare"]].apply(minmax_scale, axis=0)
titanic_embark_dummy = pd.get_dummies(titanic["Embarked"], prefix="Embarked")
improved_titanic = \
    pd.concat(objs=[titanic[["SibSp_scaled", "Parch_scaled", "Fare_scaled", "Fare"]], titanic_embark_dummy, titanic_dummies], axis=1)

# predict with the improved
lr.fit(improved_titanic[:ntitanic_train], titanic_train["Survived"])
coefficients = lr.coef_

feature_coef = pd.Series(coefficients[0], index=improved_titanic.columns)
fig = plt.figure()
feature_coef.plot.barh()

# select the features which have more than 0.7 coeff
feature_larger_point7 = feature_coef[feature_coef.abs() > 0.7].index
score = cross_validation(lr, cv=10, whole_dataset=improved_titanic[feature_larger_point7])
print("the second time prediction {:.4f}".format(score))

# select best features based on RFECV
selector = RFECV(lr, cv=10)
selector.fit(improved_titanic[:ntitanic_train], titanic_train["Survived"])
optimized = improved_titanic.columns[selector.support_]
score = cross_validation(lr, cv=10, whole_dataset=improved_titanic[optimized])
print("the third time prediciton(by RFECV) {:.4f}".format(score))

# categorize the fare
improved_titanic["Fare_category"] = pd.cut(improved_titanic["Fare"],
                                           bins=[0,12,50,100,1000], labels=["0~12", "12~50", "50~100", "100+"])
Fare_category_dummy = pd.get_dummies(improved_titanic["Fare_category"], prefix="Fare_category")
improved_titanic = pd.concat(objs=[improved_titanic, Fare_category_dummy], axis=1)
improved_titanic.drop(["Fare_scaled", "Fare_category"], axis=1, inplace=True)
lr.fit(improved_titanic[:ntitanic_train], titanic_train["Survived"])

feature_coef = pd.Series(lr.coef_[0], index=improved_titanic.columns)
feature_larger_point7 = feature_coef[feature_coef.abs() > 0.7].index
score = cross_validation(lr, cv=10, whole_dataset=improved_titanic[feature_larger_point7])
print("the fourth time prediction {:.4f}".format(score))

