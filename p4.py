import pandas as pd
import numpy as np
from scipy.optimize import minimize

def normalization_4(df,names):
  stat_dict = {}
  df_normalized = df.copy()
  for name in names:
    mean = np.array(df_normalized[[name]]).mean()
    std = np.array(df_normalized[[name]]).std()
    df_normalized[[name]] = (df_normalized[[name]]-mean)/std
    stat_dict[name] = (mean,std)

  return df_normalized, stat_dict

df_total = pd.read_csv("data.csv")

df_early = df_total[df_total["srch_booking_window"] >= 7]
df_late = df_total[df_total["srch_booking_window"] < 7]

df_early = df_early[["srch_id","prop_starrating","prop_review_score","prop_brand_bool","prop_location_score","prop_accesibility_score","prop_log_historical_price","price_usd","promotion_flag","booking_bool"]]
id_set_early = set(df_early["srch_id"])

df_late = df_late[["srch_id","prop_starrating","prop_review_score","prop_brand_bool","prop_location_score","prop_accesibility_score","prop_log_historical_price","price_usd","promotion_flag","booking_bool"]]
id_set_late = set(df_late["srch_id"])

names = ["prop_starrating","prop_review_score","prop_brand_bool","prop_location_score","prop_accesibility_score","prop_log_historical_price","price_usd","promotion_flag"]
df_normalized_late,stat_dict_late = normalization_4(df_late,names)
df_normalized_early,stat_dict_early = normalization_4(df_early,names)


def rosen_p4_late(beta):
  beta0 = beta[0]
  beta_rest = beta[1:]
  objective = 0
  for id in id_set_late:
    temp_matrix = np.array(df_normalized_late[df_normalized_late["srch_id"] == id])
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


def rosen_p4_early(beta):
  beta0 = beta[0]
  beta_rest = beta[1:]
  objective = 0
  for id in id_set_early:
    temp_matrix = np.array(df_normalized_early[df_normalized_early["srch_id"] == id])
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


beta_early = np.array([-1.74629566,  0.41212766,  0.1057882 ,  0.1008278 ,  0.02017485,
        0.04341198, -0.06984647, -1.33103003,  0.15948702])

res_early = minimize(rosen_p4_early, beta_early, method='Powell', options={'xtol': 0.0001,'disp': True})
res_early.x


beta_late = np.array([-1.74629566,  0.41212766,  0.1057882 ,  0.1008278 ,  0.02017485,
        0.04341198, -0.06984647, -1.33103003,  0.15948702])

beta_late = minimize(rosen_p4_late, beta_late, method='Powell', options={'xtol': 0.0001,'disp': True})
beta_late.x