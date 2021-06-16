import pandas as pd
import numpy as np

beta = np.array([-1.74629566,  0.41212766,  0.1057882 ,  0.1008278 ,  0.02017485,
        0.04341198, -0.06984647, -1.33103003,  0.15948702])


df = pd.read_csv("data.csv")
df1 = pd.read_csv("data1.csv")
df2 = pd.read_csv("data2.csv")
df3 = pd.read_csv("data3.csv")
df4 = pd.read_csv("data4.csv")

df1 = df1.sort_values(by=['price_usd'], inplace=False,ascending = False)
df2 = df2.sort_values(by=['price_usd'], inplace=False,ascending = False)
df3 = df3.sort_values(by=['price_usd'], inplace=False,ascending = False)
df4 = df4.sort_values(by=['price_usd'], inplace=False,ascending = False)

names = ["prop_starrating","prop_review_score","prop_brand_bool","prop_location_score","prop_accesibility_score","prop_log_historical_price","price_usd","promotion_flag"]

def normalization_p2(df,names,stat_dict):
  df_normalized = df.copy()
  for name in names:
    df_normalized[[name]]
    stat_dict[name][0]
    df_normalized[[name]] = (df_normalized[[name]]-stat_dict[name][0])/stat_dict[name][1]

  return df_normalized

def normalization_p1(df,names):
  stat_dict = {}
  df_normalized = df.copy()
  for name in names:
    mean = np.array(df_normalized[[name]]).mean()
    std = np.array(df_normalized[[name]]).std()
    df_normalized[[name]] = (df_normalized[[name]]-mean)/std
    stat_dict[name] = (mean,std)

  return df_normalized, stat_dict

df,stat_dict = normalization_p1(df,names)

df1_norm = normalization_p2(df1,names,stat_dict)
temp1 = np.array(df1_norm)
v = np.exp(temp1@beta[1:]+beta[0])
length = len(temp1)
exp_list = []
for i in range(length):
  v_temp = v[:i+1]
  temp2 = temp1[:i+1]
  p = temp2[:,-2]
  exp = (p@v_temp)/(1+np.sum(v_temp))
  exp_list.append(exp)
print(exp_list)
print(np.argmax(exp_list))


end_item = np.argmax(exp_list)
temp1 = np.array(df1_norm)[:end_item+1]
temp1 = np.exp(temp1@beta[1:]+beta[0])
price_unnormalized = np.array(df1['price_usd'])[:end_item+1]
exp_rev = (price_unnormalized@temp1)/(1+np.sum(temp1))
print("expected revenue:",exp_rev)


df2_norm = normalization_p2(df2,names,stat_dict)
temp1 = np.array(df2_norm)
v = np.exp(temp1@beta[1:]+beta[0])
length = len(temp1)
exp_list = []
for i in range(length):
  v_temp = v[:i+1]
  temp2 = temp1[:i+1]
  p = temp2[:,-2]
  exp = (p@v_temp)/(1+np.sum(v_temp))
  exp_list.append(exp)
print(exp_list)
print(np.argmax(exp_list))

end_item = np.argmax(exp_list)
temp1 = np.array(df2_norm)[:end_item+1]
temp1 = np.exp(temp1@beta[1:]+beta[0])
price_unnormalized = np.array(df2['price_usd'])[:end_item+1]
exp_rev = (price_unnormalized@temp1)/(1+np.sum(temp1))
print("expected revenue:",exp_rev)

df3_norm = normalization_p2(df3,names,stat_dict)
temp1 = np.array(df3_norm)
v = np.exp(temp1@beta[1:]+beta[0])
length = len(temp1)
exp_list = []
for i in range(length):
  v_temp = v[:i+1]
  temp2 = temp1[:i+1]
  p = temp2[:,-2]
  exp = (p@v_temp)/(1+np.sum(v_temp))
  exp_list.append(exp)
print(exp_list)
print(np.argmax(exp_list))

end_item = np.argmax(exp_list)
temp1 = np.array(df3_norm)[:end_item+1]
temp1 = np.exp(temp1@beta[1:]+beta[0])
price_unnormalized = np.array(df3['price_usd'])[:end_item+1]
exp_rev = (price_unnormalized@temp1)/(1+np.sum(temp1))
print("expected revenue:",exp_rev)

df4_norm = normalization_p2(df4,names,stat_dict)
temp1 = np.array(df4_norm)
v = np.exp(temp1@beta[1:]+beta[0])
length = len(temp1)
exp_list = []
for i in range(length):
  v_temp = v[:i+1]
  temp2 = temp1[:i+1]
  p = temp2[:,-2]
  exp = (p@v_temp)/(1+np.sum(v_temp))
  exp_list.append(exp)
print(exp_list)
print(np.argmax(exp_list))

end_item = np.argmax(exp_list)
temp1 = np.array(df4_norm)[:end_item+1]
temp1 = np.exp(temp1@beta[1:]+beta[0])
price_unnormalized = np.array(df4['price_usd'])[:end_item+1]
exp_rev = (price_unnormalized@temp1)/(1+np.sum(temp1))
print("expected revenue:",exp_rev)