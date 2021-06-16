import numpy as np
import pandas as pd
from scipy.optimize import minimize

beta = np.array([-1.74629566,  0.41212766,  0.1057882 ,  0.1008278 ,  0.02017485,
        0.04341198, -0.06984647, -1.33103003,  0.15948702])


df = pd.read_csv("data.csv")


def normalization_p1(df,names):
  stat_dict = {}
  df_normalized = df.copy()
  for name in names:
    mean = np.array(df_normalized[[name]]).mean()
    std = np.array(df_normalized[[name]]).std()
    df_normalized[[name]] = (df_normalized[[name]]-mean)/std
    stat_dict[name] = (mean,std)

  return df_normalized, stat_dict


names = ["prop_starrating","prop_review_score","prop_brand_bool","prop_location_score","prop_accesibility_score","prop_log_historical_price","price_usd","promotion_flag"]

df,stat_dict = normalization_p1(df,names)


def normalization_p3(df,names,stat_dict):
  df_normalized = df.copy()
  for name in names:
    df_normalized[[name]]
    stat_dict[name][0]
    df_normalized[[name]] = (df_normalized[[name]]-stat_dict[name][0])/stat_dict[name][1]

  return df_normalized

price_std = stat_dict["price_usd"][1]
price_mean = stat_dict["price_usd"][0]
print(stat_dict["price_usd"])


df1 = pd.read_csv("data1.csv")
df1_norm = normalization_p3(df1,names,stat_dict)

temp1 = np.array(df1_norm)
price = np.array(df1["price_usd"])
def rosen_p3(price):
  temp1[:,-2] = (price-price_mean)/price_std
  obj = -np.log(np.exp(temp1@beta[1:]+beta[0])@(price))+np.log(1+np.sum(np.exp(temp1@beta[1:]+beta[0])))
  return obj
print("price(before opt):",price)
print("rev (before opt):", np.exp(-rosen_p3(price)))
res = minimize(rosen_p3, price, method='nelder-mead', options={"maxiter":100000,'xatol': 1e-8, 'disp': True})
print("price(after opt):",res.x)
print("rev (after opt):", np.exp(-rosen_p3(res.x)))


df2 = pd.read_csv("data2.csv")
df2_norm = normalization_p3(df2,names,stat_dict)
temp1 = np.array(df2_norm)
price = np.array(df2["price_usd"])
# print("price before adjusting:",price)

def rosen_p3(price):
  temp1[:,-2] = (price-price_mean)/price_std
  obj = -np.log(np.exp(temp1@beta[1:]+beta[0])@(price))+np.log(1+np.sum(np.exp(temp1@beta[1:]+beta[0])))
  return obj
print("price(before opt):",price)
print("rev (before opt):", np.exp(-rosen_p3(price)))
res = minimize(rosen_p3, price, method='nelder-mead', options={"maxiter":100000,'xatol': 1e-8, 'disp': True})
print("price(after opt):",res.x)
print("rev (after opt):", np.exp(-rosen_p3(res.x)))


df3 = pd.read_csv("data3.csv")
df3_norm = normalization_p3(df3,names,stat_dict)
temp1 = np.array(df3_norm)
price = np.array(df3["price_usd"])
# print("price before adjusting:",price)

def rosen_p3(price):
  temp1[:,-2] = (price-price_mean)/price_std
  obj = -np.log(np.exp(temp1@beta[1:]+beta[0])@(price))+np.log(1+np.sum(np.exp(temp1@beta[1:]+beta[0])))
  return obj
print("price(before opt):",price)
print("rev (before opt):", np.exp(-rosen_p3(price)))
res = minimize(rosen_p3, price, method='nelder-mead', options={"maxiter":100000,'xatol': 1e-8, 'disp': True})
print("price(after opt):",res.x)
print("rev (after opt):", np.exp(-rosen_p3(res.x)))


df4 = pd.read_csv("data4.csv")
df4_norm = normalization_p3(df4,names,stat_dict)
temp1 = np.array(df4_norm)
price = np.array(df4["price_usd"])
# print("price before adjusting:",price)

def rosen_p3(price):
  temp1[:,-2] = (price-price_mean)/price_std
  obj = -np.log(np.exp(temp1@beta[1:]+beta[0])@(price))+np.log(1+np.sum(np.exp(temp1@beta[1:]+beta[0])))
  return obj
print("price(before opt):",price)
print("rev (before opt):", np.exp(-rosen_p3(price)))
res = minimize(rosen_p3, price, method='nelder-mead', options={"maxiter":100000,'xatol': 1e-8, 'disp': True})
print("price(after opt):",res.x)
print("rev (after opt):", np.exp(-rosen_p3(res.x)))