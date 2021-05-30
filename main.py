import copy
import os

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statlearning import plot_dist, plot_dists, plot_regressions
import warnings

warnings.simplefilter("ignore")

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train_copy = copy.deepcopy(train)
test_copy = copy.deepcopy(test)
train.head()

train_dimensions = np.shape(train)
test_dimensions = np.shape(test)


# print(test_dimensions)
# print(train_dimensions)
# print(train_copy.isna().sum())

# --------------------数据处理-------------------------#
# 填充数字的具体方法，到时候你们自己看着改
def fill_cvs(copy_name, copy_type):
    review_columns = ['review_scores_rating', 'review_scores_accuracy',
                      'review_scores_cleanliness', 'review_scores_checkin',
                      'review_scores_communication', 'review_scores_location',
                      'reviews_per_month', 'review_scores_value',
                      ]
    host_columns = ['host_is_superhost', 'host_identity_verified']
    room_columns = ['bedrooms', 'beds']
    copy_name['host_response_time'].fillna('never response', inplace=True)
    copy_name['host_response_rate'].fillna(0, inplace=True)
    copy_name['host_acceptance_rate'].fillna(0, inplace=True)
    copy_name['security_deposit'].fillna(copy_name['security_deposit'].mean(), inplace=True)
    clean_fee_median = copy_name['cleaning_fee'].median()
    copy_name['cleaning_fee'][np.isnan(copy_name['cleaning_fee'])] = clean_fee_median
    copy_name['latitude'] = np.absolute(copy_name['latitude'])
    copy_name[review_columns] = copy_name[review_columns].interpolate(limit_direction='backward')
    copy_name[host_columns] = copy_name[host_columns].fillna('f')
    for item in room_columns:
        copy_name[item].fillna(copy_name[item].mean(), inplace=True)
    if copy_type == 'train':
        copy_name = copy_name.dropna()

    return copy_name


train_copy = fill_cvs(train_copy, 'train')
test_copy = fill_cvs(test_copy, 'test')
print(test_copy['latitude'])
# print(train_copy.isna().sum())
# print('----------------')
# print(test_copy.isna().sum())
# print(np.shape(train_copy))
# print(np.shape(test_copy))

train_copy_for_fe = copy.deepcopy(train_copy)
test_copy_for_fe = copy.deepcopy(test_copy)


# 文本，boolean的数据类型量化
def fe_cvs(fe_copy):
    response_time_map = {'within an hour': 4,
                         'within a few hours': 3,
                         'within a day': 2,
                         'a few days or more': 1,
                         'never response': 0}

    cancellation_policy_map = {
        'flexible': 4,
        'moderate': 3,
        'strict_14_with_grace_period': 2,
        'super_strict_30': 1,
        'super_strict_60': 0
    }

    boolean2int_map = {'t': 1,
                       'f': 0}
    boolean2int_list = ['host_identity_verified', 'instant_bookable', 'require_guest_profile_picture',
                        'require_guest_phone_verification', 'host_is_superhost']
    fe_copy['host_response_time'] = fe_copy['host_response_time'].map(response_time_map)
    fe_copy['cancellation_policy'] = fe_copy['cancellation_policy'].map(cancellation_policy_map)
    for item in boolean2int_list:
        fe_copy[item] = fe_copy[item].map(boolean2int_map)

    return fe_copy


train_copy_for_fe = fe_cvs(train_copy_for_fe)
test_copy_for_fe = fe_cvs(test_copy_for_fe)


# print(train_copy_for_fe.isna().sum())
# print(test_copy_for_fe.isna().sum())

# print(pd.crosstab(index=train_copy_for_fe['property_type'], columns='count'))
# print(pd.crosstab(index=train_copy_for_fe['room_type'], columns='count'))

# 对数据量比较少的种类合并
def merge_redundant(fe_copy):
    fe_copy['property_type'] = np.where(fe_copy['property_type'].str.contains('House'), 'House',
                                        (np.where(fe_copy['property_type'].str.contains('Apartment'), 'Apartment',
                                                  np.where(fe_copy['property_type'].str.contains('Townhouse'),
                                                           'Townhouse', 'Other type')))
                                        )

    fe_copy['room_type'] = np.where(fe_copy['room_type'].str.contains('Private room'), 'Private room',
                                    (np.where(fe_copy['room_type'].str.contains('Entire home/apt'),
                                              'Entire home/apt', 'Other type')))

    return fe_copy


train_copy_for_fe = merge_redundant(train_copy_for_fe)
test_copy_for_fe = merge_redundant(test_copy_for_fe)


# print(pd.crosstab(index=train_copy_for_fe['property_type'], columns='count'))
# print(pd.crosstab(index=train_copy_for_fe['room_type'], columns='count'))
def dummy_create(dm_copy):
    dm_copy = copy.deepcopy(pd.get_dummies(dm_copy, drop_first=True))
    # print(dm_copy.head)
    return dm_copy


test_copy_for_dm = dummy_create(test_copy_for_fe)
train_copy_for_dm = dummy_create(train_copy_for_fe)

# -----------分析数据------------#
"""

根据数据类型种类分为两个组：
如果只有0 1在dummy里，如果很复杂就在other里

"""
# values 0 or 1
dummy_list = []
# other values type
other_list = []

for item in train_copy_for_dm.columns:
    # print(item, '---', train_copy_for_dm[item].value_counts().shape[0])
    if item == 'Id':
        continue
    if train_copy_for_dm[item].value_counts().shape[0] > 2:
        other_list.append(item)
    else:
        dummy_list.append(item)
# print(dummy_list)
# print(other_list)

train_des = train_copy_for_fe.describe()
train_des.loc['skew', :] = train_copy_for_dm.skew()
train_des.loc['kurt', :] = train_copy_for_dm.kurt()
train_des[other_list].round(3)

# ------------画图---------------#
# log_y_train = np.log(train_copy_for_dm['price'])
# plot_dist(log_y_train)
# plt.title('tmp')
# plt.show()
#
# plot_dists(train_copy_for_dm[other_list])
# plt.show()
# reg_other = copy.deepcopy(other_list)
# reg_other.remove('price')
# plot_regressions(train_copy_for_dm[reg_other], train_copy_for_dm['price'])
# plt.show()
#
# sns.boxplot(x=train_copy_for_dm.loc[:, 'bed_type_Real Bed'], y=train_copy_for_dm.loc[:, 'price'], palette='Blues')
# sns.despine()
# plt.show()
#
# # 对 'host_identity_verified', 'host_response_time','cancellation_policy' 做同样的图
#
# rows = train_copy_for_dm['accommodates'] <= 12
# sns.boxplot(x=train_copy_for_dm.loc[rows, 'accommodates'], y=train_copy_for_dm.loc[rows, 'price'], palette='Blues')
# sns.despine()
# plt.show()
#
# corr_map = train_copy_for_dm.corr()['price'].sort_values()
# print(corr_map)
# COR_THRESHOLD = 0.08
# train_cor = train_copy_for_dm[corr_map.loc[(corr_map > COR_THRESHOLD) | (corr_map < -COR_THRESHOLD)].index]
# plt.subplots(figsize=(20, 15))
# sns.heatmap(train_cor.corr(), square=True, annot=True, cmap="Accent")
# plt.title("heap map")
# plt.show()
#
# # 删除异常值，自己看前面的图，删一些离谱的数据
# train_copy_for_dm = train_copy_for_dm[-((train_copy_for_dm['extra_people'] > 100) |
#                                         (train_copy_for_dm['security_deposit'] > 2000))]
# print(train_copy_for_dm.shape)

# -------------------------#

x_train = train_copy_for_dm.drop(['price', 'Id'], axis=1)
y_train = train_copy_for_dm['price']
x_test = test_copy_for_dm.drop(['Id'], axis=1)

log_train = copy.deepcopy(train_copy_for_dm)
log_test = copy.deepcopy(test_copy_for_dm)

des = log_train.describe()
des.loc['skew', :] = log_train.skew()
des.loc['kurt', :] = log_train.kurt()
des = des[other_list].T
# print(des)

