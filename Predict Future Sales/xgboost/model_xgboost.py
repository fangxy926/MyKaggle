import numpy as np
import pandas as pd

from itertools import product
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from xgboost import plot_importance
import gc
import pickle
from sklearn.preprocessing import LabelEncoder


def plot_features(booster, figsize):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    return plot_importance(booster=booster, ax=ax)


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)

if __name__ == '__main__':
    # Data Cleaning
    train = pd.read_csv('../data/sales_train.csv', low_memory=False)
    test = pd.read_csv('../data/test.csv', low_memory=False).set_index('ID')
    shops = pd.read_csv('../data/shops.csv', low_memory=False)
    items = pd.read_csv('../data/items.csv', low_memory=False)
    cats = pd.read_csv('../data/item_categories.csv', low_memory=False)

    train = train[train.item_cnt_day <= 1000]
    train = train[train.item_price < 100000]

    # 计算数据的中位数
    median = train[(train.shop_id == 32) & (train.item_id == 2973) & (train.date_block_num == 4) & (
            train.item_price > 0)].item_price.median()

    # 用商品价格的中位数去替换商品价格小于0的商品记录
    train.loc[train.item_price < 0, 'item_price'] = median
    # 有几个商店的名称更其他商店重复了需要修改，0和57、1和58、10和11
    # Якутск Орджоникидзе, 56
    train.loc[train.shop_id == 0, 'shop_id'] = 57
    test.loc[test.shop_id == 0, 'shop_id'] = 57
    # Якутск ТЦ "Центральный"
    train.loc[train.shop_id == 1, 'shop_id'] = 58
    test.loc[test.shop_id == 1, 'shop_id'] = 58
    # Жуковский ул. Чкалова 39м²
    train.loc[train.shop_id == 10, 'shop_id'] = 11
    test.loc[test.shop_id == 10, 'shop_id'] = 11

    # 从shop name中提取出city信息
    shops.loc[shops.shop_name == 'Сергиев Посад ТЦ "7Я"', 'shop_name'] = 'СергиевПосад ТЦ "7Я"'
    shops['city'] = shops['shop_name'].str.split(' ').map(lambda x: x[0])
    shops.loc[shops.city == '!Якутск', 'city'] = 'Якутск'
    shops['city_code'] = LabelEncoder().fit_transform(shops['city'])
    shops = shops[['shop_id', 'city_code']]

    # 商品类别是每个商品的小类别，在category_name里还可分出两种大类别，作为两个新的特征type和subtype。
    # 有两种情况，一种第一个有‘-’分割成两个元素第一个元素就是首选类别，第二个元素就是代替类别；
    # 没有‘-’则首选类别和代替类别都一样。并将这两个特征因子化。
    cats['split'] = cats['item_category_name'].str.split('-')
    cats['type'] = cats['split'].map(lambda x: x[0].strip())
    cats['type_code'] = LabelEncoder().fit_transform(cats['type'])
    # if subtype is nan then type
    cats['subtype'] = cats['split'].map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())
    cats['subtype_code'] = LabelEncoder().fit_transform(cats['subtype'])
    cats = cats[['item_category_id', 'type_code', 'subtype_code']]

    items.drop(['item_name'], axis=1, inplace=True)

    # 建立一个新的数据集matrix里面包含三个特征：date_block_num、shop_id、item_id,三个特征数据来源于训练集且三个特征都不重复
    matrix = []
    cols = ['date_block_num', 'shop_id', 'item_id']
    for i in range(34):
        sales = train[train.date_block_num == i]
        matrix.append(np.array(list(product([i], sales.shop_id.unique(), sales.item_id.unique())), dtype='int16'))
    matrix = pd.DataFrame(np.vstack(matrix), columns=cols)  # np.vstack()沿着竖直方向将矩阵堆叠起来
    matrix['date_block_num'] = matrix['date_block_num'].astype(np.int8)  # 转类型
    matrix['shop_id'] = matrix['shop_id'].astype(np.int8)
    matrix['item_id'] = matrix['item_id'].astype(np.int16)
    matrix.sort_values(cols, inplace=True)

    train['revenue'] = train['item_price'] * train['item_cnt_day']  # 某商品的收入，在训练集后多加一行revenue

    # 将某时间块内某商店的某商品作为key，可以求得每个月的销售量
    group = train.groupby(['date_block_num', 'shop_id', 'item_id']).agg({'item_cnt_day': ['sum']})
    group.columns = ['item_cnt_month']
    group.reset_index(inplace=True)
    matrix = pd.merge(matrix, group, on=cols, how='left')
    matrix['item_cnt_month'] = (matrix['item_cnt_month']
                                .fillna(0)
                                .clip(0, 20)  # NB clip target here
                                .astype(np.float16))

    # 对测试集的数据进行处理
    test['date_block_num'] = 34
    test['date_block_num'] = test['date_block_num'].astype(np.int8)
    test['shop_id'] = test['shop_id'].astype(np.int8)
    test['item_id'] = test['item_id'].astype(np.int16)

    # 将matrix和test两张表首尾连接到matrix
    matrix = pd.concat([matrix, test], ignore_index=True, sort=False, keys=cols)
    matrix.fillna(0, inplace=True)  # 34 month

    # left join
    matrix = pd.merge(matrix, shops, on=['shop_id'], how='left')
    matrix = pd.merge(matrix, items, on=['item_id'], how='left')
    matrix = pd.merge(matrix, cats, on=['item_category_id'], how='left')
    matrix['city_code'] = matrix['city_code'].astype(np.int8)
    matrix['item_category_id'] = matrix['item_category_id'].astype(np.int8)
    matrix['type_code'] = matrix['type_code'].astype(np.int8)
    matrix['subtype_code'] = matrix['subtype_code'].astype(np.int8)


    # lag,避免了一个个的取填充
    def lag_feature(df, lags, col):
        tmp = df[['date_block_num', 'shop_id', 'item_id', col]]
        for i in lags:
            shifted = tmp.copy()
            shifted.columns = ['date_block_num', 'shop_id', 'item_id', col + '_lag_' + str(i)]
            shifted['date_block_num'] += i
            df = pd.merge(df, shifted, on=['date_block_num', 'shop_id', 'item_id'], how='left')
        return df


    # 构造商店和单品两两配对后在1，2，3，6，12月后item_cnt_month的特征值
    matrix = lag_feature(matrix, [1, 2, 3, 6, 12], 'item_cnt_month')

    # 构造在过了一月后item_cnt_month的平均值的特征值
    group = matrix.groupby(['date_block_num']).agg({'item_cnt_month': ['mean']})
    group.columns = ['date_avg_item_cnt']
    group.reset_index(inplace=True)

    matrix = pd.merge(matrix, group, on=['date_block_num'], how='left')
    matrix['date_avg_item_cnt'] = matrix['date_avg_item_cnt'].astype(np.float16)
    matrix = lag_feature(matrix, [1], 'date_avg_item_cnt')
    matrix.drop(['date_avg_item_cnt'], axis=1, inplace=True)

    # 不重复单品和月份在过了1，2，3，6，12月后item_cnt_month的平均值的特征值
    group = matrix.groupby(['date_block_num', 'item_id']).agg({'item_cnt_month': ['mean']})
    group.columns = ['date_item_avg_item_cnt']
    group.reset_index(inplace=True)

    matrix = pd.merge(matrix, group, on=['date_block_num', 'item_id'], how='left')
    matrix['date_item_avg_item_cnt'] = matrix['date_item_avg_item_cnt'].astype(np.float16)
    matrix = lag_feature(matrix, [1, 2, 3, 6, 12], 'date_item_avg_item_cnt')
    matrix.drop(['date_item_avg_item_cnt'], axis=1, inplace=True)

    # 不同的商店和月份在过了1，2，3，6，12月后item_cnt_month的平均值的特征值
    group = matrix.groupby(['date_block_num', 'shop_id']).agg({'item_cnt_month': ['mean']})
    group.columns = ['date_shop_avg_item_cnt']
    group.reset_index(inplace=True)

    matrix = pd.merge(matrix, group, on=['date_block_num', 'shop_id'], how='left')
    matrix['date_shop_avg_item_cnt'] = matrix['date_shop_avg_item_cnt'].astype(np.float16)
    matrix = lag_feature(matrix, [1, 2, 3, 6, 12], 'date_shop_avg_item_cnt')
    matrix.drop(['date_shop_avg_item_cnt'], axis=1, inplace=True)

    # 不同的item_category_id和月份在过了1月后item_cnt_month的平均值的特征值
    group = matrix.groupby(['date_block_num', 'item_category_id']).agg({'item_cnt_month': ['mean']})
    group.columns = ['date_cat_avg_item_cnt']
    group.reset_index(inplace=True)

    matrix = pd.merge(matrix, group, on=['date_block_num', 'item_category_id'], how='left')
    matrix['date_cat_avg_item_cnt'] = matrix['date_cat_avg_item_cnt'].astype(np.float16)
    matrix = lag_feature(matrix, [1], 'date_cat_avg_item_cnt')
    matrix.drop(['date_cat_avg_item_cnt'], axis=1, inplace=True)

    # 不同的item_category_id和不同的商店、月份在过了1月后item_cnt_month的平均值的特征值
    group = matrix.groupby(['date_block_num', 'shop_id', 'item_category_id']).agg({'item_cnt_month': ['mean']})
    group.columns = ['date_shop_cat_avg_item_cnt']
    group.reset_index(inplace=True)

    matrix = pd.merge(matrix, group, on=['date_block_num', 'shop_id', 'item_category_id'], how='left')
    matrix['date_shop_cat_avg_item_cnt'] = matrix['date_shop_cat_avg_item_cnt'].astype(np.float16)
    matrix = lag_feature(matrix, [1], 'date_shop_cat_avg_item_cnt')
    matrix.drop(['date_shop_cat_avg_item_cnt'], axis=1, inplace=True)

    # 不同的type_code和不同的商店、月份在过了1月后item_cnt_month的平均值的特征值
    group = matrix.groupby(['date_block_num', 'shop_id', 'type_code']).agg({'item_cnt_month': ['mean']})
    group.columns = ['date_shop_type_avg_item_cnt']
    group.reset_index(inplace=True)

    matrix = pd.merge(matrix, group, on=['date_block_num', 'shop_id', 'type_code'], how='left')
    matrix['date_shop_type_avg_item_cnt'] = matrix['date_shop_type_avg_item_cnt'].astype(np.float16)
    matrix = lag_feature(matrix, [1], 'date_shop_type_avg_item_cnt')
    matrix.drop(['date_shop_type_avg_item_cnt'], axis=1, inplace=True)

    # 不同的subtype_code和不同的商店、月份在过了1月后item_cnt_month的平均值的特征值
    group = matrix.groupby(['date_block_num', 'shop_id', 'subtype_code']).agg({'item_cnt_month': ['mean']})
    group.columns = ['date_shop_subtype_avg_item_cnt']
    group.reset_index(inplace=True)

    matrix = pd.merge(matrix, group, on=['date_block_num', 'shop_id', 'subtype_code'], how='left')
    matrix['date_shop_subtype_avg_item_cnt'] = matrix['date_shop_subtype_avg_item_cnt'].astype(np.float16)
    matrix = lag_feature(matrix, [1], 'date_shop_subtype_avg_item_cnt')
    matrix.drop(['date_shop_subtype_avg_item_cnt'], axis=1, inplace=True)

    # 不同的citycode和月份在过了1月后item_cnt_month的平均值的特征值
    group = matrix.groupby(['date_block_num', 'city_code']).agg({'item_cnt_month': ['mean']})
    group.columns = ['date_city_avg_item_cnt']
    group.reset_index(inplace=True)

    matrix = pd.merge(matrix, group, on=['date_block_num', 'city_code'], how='left')
    matrix['date_city_avg_item_cnt'] = matrix['date_city_avg_item_cnt'].astype(np.float16)
    matrix = lag_feature(matrix, [1], 'date_city_avg_item_cnt')
    matrix.drop(['date_city_avg_item_cnt'], axis=1, inplace=True)

    # 不同的citycode和不重复单品、月份在过了1月后item_cnt_month的平均值的特征值
    group = matrix.groupby(['date_block_num', 'item_id', 'city_code']).agg({'item_cnt_month': ['mean']})
    group.columns = ['date_item_city_avg_item_cnt']
    group.reset_index(inplace=True)

    matrix = pd.merge(matrix, group, on=['date_block_num', 'item_id', 'city_code'], how='left')
    matrix['date_item_city_avg_item_cnt'] = matrix['date_item_city_avg_item_cnt'].astype(np.float16)
    matrix = lag_feature(matrix, [1], 'date_item_city_avg_item_cnt')
    matrix.drop(['date_item_city_avg_item_cnt'], axis=1, inplace=True)

    # 不同的type_code和月份在过了1月后item_cnt_month的平均值的特征值
    group = matrix.groupby(['date_block_num', 'type_code']).agg({'item_cnt_month': ['mean']})
    group.columns = ['date_type_avg_item_cnt']
    group.reset_index(inplace=True)

    matrix = pd.merge(matrix, group, on=['date_block_num', 'type_code'], how='left')
    matrix['date_type_avg_item_cnt'] = matrix['date_type_avg_item_cnt'].astype(np.float16)
    matrix = lag_feature(matrix, [1], 'date_type_avg_item_cnt')
    matrix.drop(['date_type_avg_item_cnt'], axis=1, inplace=True)

    # 不同的subtype_code和月份在过了1月后item_cnt_month的平均值的特征值
    group = matrix.groupby(['date_block_num', 'subtype_code']).agg({'item_cnt_month': ['mean']})
    group.columns = ['date_subtype_avg_item_cnt']
    group.reset_index(inplace=True)

    matrix = pd.merge(matrix, group, on=['date_block_num', 'subtype_code'], how='left')
    matrix['date_subtype_avg_item_cnt'] = matrix['date_subtype_avg_item_cnt'].astype(np.float16)
    matrix = lag_feature(matrix, [1], 'date_subtype_avg_item_cnt')
    matrix.drop(['date_subtype_avg_item_cnt'], axis=1, inplace=True)

    # 对不重复单品算出它的价格均值item_avg_item_price，然后leftjoin入matrix数据集中
    group = train.groupby(['item_id']).agg({'item_price': ['mean']})
    group.columns = ['item_avg_item_price']
    group.reset_index(inplace=True)

    matrix = pd.merge(matrix, group, on=['item_id'], how='left')
    matrix['item_avg_item_price'] = matrix['item_avg_item_price'].astype(np.float16)

    # 对不同月份的不重复单品算出它的价格均值date_item_avg_item_price，然后leftjoin入matrix数据集中
    group = train.groupby(['date_block_num', 'item_id']).agg({'item_price': ['mean']})
    group.columns = ['date_item_avg_item_price']
    group.reset_index(inplace=True)

    matrix = pd.merge(matrix, group, on=['date_block_num', 'item_id'], how='left')
    matrix['date_item_avg_item_price'] = matrix['date_item_avg_item_price'].astype(np.float16)

    # 构造date_item_avg_item_price构造1，2，3，4，5，6 月后的date_item_avg_item_price特征值
    lags = [1, 2, 3, 4, 5, 6]
    matrix = lag_feature(matrix, lags, 'date_item_avg_item_price')

    for i in lags:
        matrix['delta_price_lag_' + str(i)] = \
            (matrix['date_item_avg_item_price_lag_' + str(i)] - matrix['item_avg_item_price']) / matrix[
                'item_avg_item_price']


    def select_trend(row):
        for i in lags:
            if row['delta_price_lag_' + str(i)]:
                return row['delta_price_lag_' + str(i)]
        return 0


    matrix['delta_price_lag'] = matrix.apply(select_trend, axis=1)
    matrix['delta_price_lag'] = matrix['delta_price_lag'].astype(np.float16)
    matrix['delta_price_lag'].fillna(0, inplace=True)

    fetures_to_drop = ['item_avg_item_price', 'date_item_avg_item_price']
    for i in lags:
        fetures_to_drop += ['date_item_avg_item_price_lag_' + str(i)]
        fetures_to_drop += ['delta_price_lag_' + str(i)]

    matrix.drop(fetures_to_drop, axis=1, inplace=True)

    # 通过每天的revenue求出每月不同商店的revenue总和date_shop_revenue
    group = train.groupby(['date_block_num', 'shop_id']).agg({'revenue': ['sum']})
    group.columns = ['date_shop_revenue']
    group.reset_index(inplace=True)

    matrix = pd.merge(matrix, group, on=['date_block_num', 'shop_id'], how='left')
    matrix['date_shop_revenue'] = matrix['date_shop_revenue'].astype(np.float32)

    # 通过date_shop_revenue求出不同的商店在所有月份的均值shop_avg_revenue
    group = group.groupby(['shop_id']).agg({'date_shop_revenue': ['mean']})
    group.columns = ['shop_avg_revenue']
    group.reset_index(inplace=True)

    matrix = pd.merge(matrix, group, on=['shop_id'], how='left')
    matrix['shop_avg_revenue'] = matrix['shop_avg_revenue'].astype(np.float32)

    # delta_revenue收益趋势为date_shop_revenue减去shop_avg_revenue除以shop_avg_revenue，然后构造过了一月后的趋势
    matrix['delta_revenue'] = (matrix['date_shop_revenue'] - matrix['shop_avg_revenue']) / matrix['shop_avg_revenue']
    matrix['delta_revenue'] = matrix['delta_revenue'].astype(np.float16)

    matrix = lag_feature(matrix, [1], 'delta_revenue')

    matrix.drop(['date_shop_revenue', 'shop_avg_revenue', 'delta_revenue'], axis=1, inplace=True)

    # 求出每一行对应月份month，然后转换为天数获得特征days
    matrix['month'] = matrix['date_block_num'] % 12
    days = pd.Series([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    matrix['days'] = matrix['month'].map(days).astype(np.int8)

    # 通过缓存机制构造特殊特征item_shop_last_sale
    cache = {}
    matrix['item_shop_last_sale'] = -1
    matrix['item_shop_last_sale'] = matrix['item_shop_last_sale'].astype(np.int8)
    for idx, row in matrix.iterrows():
        key = str(row.item_id) + ' ' + str(row.shop_id)
        if key not in cache:
            if row.item_cnt_month != 0:
                cache[key] = row.date_block_num
        else:
            last_date_block_num = cache[key]
            matrix.at[idx, 'item_shop_last_sale'] = row.date_block_num - last_date_block_num
            cache[key] = row.date_block_num

    # item_last_sale
    cache = {}
    matrix['item_last_sale'] = -1
    matrix['item_last_sale'] = matrix['item_last_sale'].astype(np.int8)
    for idx, row in matrix.iterrows():  # DataFrame的遍历函数
        key = row.item_id
        if key not in cache:
            if row.item_cnt_month != 0:
                cache[key] = row.date_block_num
        else:
            last_date_block_num = cache[key]
            if row.date_block_num > last_date_block_num:
                matrix.at[idx, 'item_last_sale'] = row.date_block_num - last_date_block_num
                cache[key] = row.date_block_num

    # 构造某次与第一次售出的时间间隔item_shop_first_sale和item_first_sale
    matrix['item_shop_first_sale'] = matrix['date_block_num'] - matrix.groupby(['item_id', 'shop_id'])[
        'date_block_num'].transform('min')
    matrix['item_first_sale'] = matrix['date_block_num'] - matrix.groupby('item_id')['date_block_num'].transform('min')

    # prepare
    matrix = matrix[matrix.date_block_num > 11]


    def fill_na(df):
        for col in df.columns:
            if ('_lag_' in col) & (df[col].isnull().any()):
                if ('item_cnt' in col):
                    df[col].fillna(0, inplace=True)
        return df


    matrix = fill_na(matrix)
    matrix.to_pickle('data.pkl')  # 保存数据为pkl文件
    del matrix
    del cache
    del group
    del items
    del shops
    del cats
    del train
    # leave test for submission
    gc.collect();

    data = pd.read_pickle('data.pkl')

    data = data[[
        'date_block_num',
        'shop_id',
        'item_id',
        'item_cnt_month',
        'city_code',
        'item_category_id',
        'type_code',
        'subtype_code',
        'item_cnt_month_lag_1',
        'item_cnt_month_lag_2',
        'item_cnt_month_lag_3',
        'item_cnt_month_lag_6',
        'item_cnt_month_lag_12',
        'date_avg_item_cnt_lag_1',
        'date_item_avg_item_cnt_lag_1',
        'date_item_avg_item_cnt_lag_2',
        'date_item_avg_item_cnt_lag_3',
        'date_item_avg_item_cnt_lag_6',
        'date_item_avg_item_cnt_lag_12',
        'date_shop_avg_item_cnt_lag_1',
        'date_shop_avg_item_cnt_lag_2',
        'date_shop_avg_item_cnt_lag_3',
        'date_shop_avg_item_cnt_lag_6',
        'date_shop_avg_item_cnt_lag_12',
        'date_cat_avg_item_cnt_lag_1',
        'date_shop_cat_avg_item_cnt_lag_1',
        # 'date_shop_type_avg_item_cnt_lag_1',
        # 'date_shop_subtype_avg_item_cnt_lag_1',
        'date_city_avg_item_cnt_lag_1',
        'date_item_city_avg_item_cnt_lag_1',
        # 'date_type_avg_item_cnt_lag_1',
        # 'date_subtype_avg_item_cnt_lag_1',
        'delta_price_lag',
        'month',
        'days',
        'item_shop_last_sale',
        'item_last_sale',
        'item_shop_first_sale',
        'item_first_sale',
    ]]

    X_train = data[data.date_block_num < 33].drop(['item_cnt_month'], axis=1)
    Y_train = data[data.date_block_num < 33]['item_cnt_month']
    X_valid = data[data.date_block_num == 33].drop(['item_cnt_month'], axis=1)
    Y_valid = data[data.date_block_num == 33]['item_cnt_month']
    X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)

    del data
    gc.collect();

    model = XGBRegressor(
        max_depth=8,  # 树的最大深度,防止过拟合
        n_estimators=1000,
        min_child_weight=300,  # 最小叶子节点样本权重和
        colsample_bytree=0.8,  # 控制每棵随机采样的列数的占比
        subsample=0.8,  # 控制对于每棵树，随机采样的比例
        eta=0.3,  # learning rate
        seed=42)

    model.fit(
        X_train,
        Y_train,
        eval_metric="rmse",
        eval_set=[(X_train, Y_train), (X_valid, Y_valid)],
        verbose=True,
        early_stopping_rounds=10)

    Y_pred = model.predict(X_valid).clip(0, 20)
    Y_test = model.predict(X_test).clip(0, 20)

    submission = pd.DataFrame({
        "ID": test.index,
        "item_cnt_month": Y_test
    })
    submission.to_csv('xgb_submission.csv', index=False)

    # save predictions for an ensemble
    pickle.dump(Y_pred, open('xgb_train.pickle', 'wb'))
    pickle.dump(Y_test, open('xgb_test.pickle', 'wb'))

    plot_features(model, (10, 14))
