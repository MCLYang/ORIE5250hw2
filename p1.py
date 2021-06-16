import pandas as pd
import numpy as np
from scipy.optimize import minimize

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


df = df[["srch_id","prop_starrating","prop_review_score","prop_brand_bool","prop_location_score","prop_accesibility_score","prop_log_historical_price","price_usd","promotion_flag","booking_bool"]]
id_set = set(df["srch_id"])

names = ["prop_starrating","prop_review_score","prop_brand_bool","prop_location_score","prop_accesibility_score","prop_log_historical_price","price_usd","promotion_flag"]
df_normalized,stat_dict = normalization(df,names)

def rosen(beta):
  beta0 = beta[0]
  beta_rest = beta[1:]
  objective = 0
  for id in id_set:
    temp_matrix = np.array(df_normalized[df_normalized["srch_id"] == id])
    j = np.where(temp_matrix[:,-1] == 1)[0]      
    v_j = temp_matrix[j,1:-1]
    v_p = temp_matrix[:,1:-1]
    if len(v_j) != 0:
      linear = beta0+v_j@beta_rest 
    else:
      linear = 0 
    concave = np.log(np.sum(np.exp(beta0 + v_p@beta_rest))+1)
    T = linear-concave
    # print("T:",T)
    objective = objective + T
  #max -> min
  objective = -objective
  print("objective:",objective)
  print(beta)
  return objective

beta = np.zeros(9)
res = minimize(rosen, beta, method='Powell', options={'xtol': 1e-8, 'disp': True})
beta = res.x
res.x