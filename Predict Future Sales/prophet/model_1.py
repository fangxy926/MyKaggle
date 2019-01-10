import pandas as pd
import time
from fbprophet import Prophet

# 导入数据
sales = pd.read_csv("./data/sales_train.csv")
test = pd.read_csv("./data/test.csv")
item_cat = pd.read_csv("./data/item_categories.csv")
item = pd.read_csv("./data/items.csv")
sub = pd.read_csv("./data/sample_submission.csv")
shops = pd.read_csv("./data/shops.csv")

sales = sales[(sales.item_price < 100000) & (sales.item_cnt_day < 1001)]

median = sales[(sales.shop_id == 32) & (sales.item_id == 2973) & (sales.date_block_num == 4) &
               (sales.item_price > 0)].item_price.median()
sales.loc[sales.item_price <= 0, 'item_price'] = median

monthly_sales = sales.groupby(["shop_id", "item_id", "date_block_num"])["item_cnt_day"].sum()
monthly_sales = monthly_sales.unstack(level=-1).fillna(0)
monthly_sales = monthly_sales.T
dates = pd.date_range(start='2013-01-01', end='2015-10-01', freq='MS')
monthly_sales.index = dates
# monthly_sales=monthly_sales.reset_index()
monthly_sales.head()

start_time = time.time()

forecastsDict = {}

for i in range(len(test)):
    shop_id = test.iloc[i]['shop_id']
    item_id = test.iloc[i]['item_id']

    train_shop = sales[sales['shop_id'] == shop_id]
    train_item = train_shop[train_shop['item_id'] == item_id]

    if not train_item.empty:
        nodeToForecast = pd.DataFrame(monthly_sales[shop_id][item_id])
        nodeToForecast.reset_index(inplace=True)
        nodeToForecast = nodeToForecast.rename(
            columns={nodeToForecast.columns[0]: 'ds', nodeToForecast.columns[1]: 'y'})

        growth = 'linear'
        m = Prophet(growth, yearly_seasonality=True)
        m.fit(nodeToForecast)
        future = m.make_future_dataframe(periods=1, freq='MS')
        forecast = m.predict(future)
        yhat = forecast['yhat'].iloc[-1]
        if yhat < 0:
            yhat = 0
        forecastsDict[i] = yhat
    else:
        forecastsDict[i] = 0
    print('#' + str(i) + ': ' + str(forecastsDict[i]))

end_time = time.time()

submission = pd.DataFrame({'ID': test['ID'], 'item_cnt_month': forecastsDict})
submission.to_csv('./submission.csv', index=False, sep=',')
