import pandas as pd
import numpy as np
from scipy.optimize import minimize
import gurobipy as gp
from gurobipy import GRB
from itertools import product
import pdb

#results from P1-P4
#================================
stat_dict = {'price_usd': (141.19362259736354, 181.7693498769592),
 'promotion_flag': (0.14480847531844532, 0.3519076310545035),
 'prop_accesibility_score': (0.005993111516316034, 0.07718286163824864),
 'prop_brand_bool': (0.7357279637145527, 0.44094481187898166),
 'prop_location_score': (2.6171728460417363, 1.3481508978402192),
 'prop_log_historical_price': (4.27631054382422, 1.7895033851701312),
 'prop_review_score': (3.9876804632407246, 0.9071191629160908),
 'prop_starrating': (3.157402505734957, 0.8571822862649962)}
normalization_colomns = ["prop_starrating","prop_review_score","prop_brand_bool","prop_location_score","prop_accesibility_score","prop_log_historical_price","price_usd","promotion_flag"]
late_beta = np.array([-1.53953067,  0.46527555,  0.09313614 , 0.11243684, 0.08595676,   0.02554587,   -0.03388866,   -1.6961064,0.19450294])
late_theta = 0.45690687096001914
early_beta = np.array([-1.9180075,  0.37792879,  0.12591464,  0.09239193, -0.02085044,  0.05785425,  -0.09397025,  -1.07208896,0.13451358])
early_theta = 0.5430931290399809

#early first
theta = np.array([early_theta,late_theta])
beta = np.array([early_beta,late_beta])

#================================


def normalization(df,names,stat_dict):
  df_normalized = df.copy()
  for name in names:
    df_normalized[[name]]
    stat_dict[name][0]
    df_normalized[[name]] = (df_normalized[[name]]-stat_dict[name][0])/stat_dict[name][1]

  return df_normalized

df1 = pd.read_csv("data1.csv")
df2 = pd.read_csv("data2.csv")
df3 = pd.read_csv("data3.csv")
df4 = pd.read_csv("data4.csv")
df1 = df1.sort_values(by=['price_usd'], inplace=False,ascending = False)
df2 = df2.sort_values(by=['price_usd'], inplace=False,ascending = False)
df3 = df3.sort_values(by=['price_usd'], inplace=False,ascending = False)
df4 = df4.sort_values(by=['price_usd'], inplace=False,ascending = False)


def return_Revenue_mix(beta,theta,canidates,price,normalized_data):
  num_item = len(canidates)
  num_groups = len(beta)
  temp_data = normalized_data[canidates]
  temp_price = price[canidates]

  V = []
  for k in range(num_groups):
    V.append(np.exp(temp_data@beta[k,1:]+beta[k,0]))
  V = np.array(V).transpose()

  r = 0
  for k,b in enumerate(beta):
    r = r + theta[k]*(temp_price@V[:,k])/(1+np.sum(V[:,k]))
    
  return r

def return_canidate(beta,normalized_data):
  v = np.exp(normalized_data@beta[1:]+beta[0])
  exp_list = []
  temp1 = np.array(normalized_data)
  length = len(temp1)
  for i in range(length):  
    v_temp = v[:i+1]
    temp2 = normalized_data[:i+1]
    temp2 = temp1[:i+1]
    p = temp2[:,-2]
    exp = (p@v_temp)/(1+np.sum(v_temp))
    exp_list.append(exp)
  c = np.argmax(exp_list)
  # pdb.set_trace()
  return(np.arange(c+1))

def return_Revenue(beta,canidates,price,normalized_data):
  num_item = len(canidates)
  num_groups = len(beta)
  temp_data = normalized_data[canidates]
  temp_price = price[canidates]

  V = np.exp(temp_data@beta[1:]+beta[0])
  exp_rev = (temp_price@V)/(1+np.sum(V))
    
  return exp_rev


def get_model(data_np,price,M,num_groups = 2):
  m2 = gp.Model('MIP')
  num_products = len(data_np)
  cartesian_prod = list(product(range(num_products),range(num_groups)))
  
  V = []
  for k in range(num_groups):
    V.append(np.exp(data_np@beta[k,1:]+beta[k,0]))

  V = np.array(V).transpose()

  X = m2.addVars(num_products, vtype=GRB.BINARY, name='X')
  Y = m2.addVars(cartesian_prod, vtype=GRB.CONTINUOUS, name='Y')
  Z = m2.addVars(num_groups, vtype=GRB.CONTINUOUS, name='Z')


  m2.update()
  
  m2.setObjective(-gp.quicksum(theta[k]*Z[k] for k in range(num_groups)), GRB.MINIMIZE)
  m2.update()

  # m2.addConstrs(Z[k] == ((np.multiply(price,X)@(np.exp(data_np@beta[k,1:]+beta[k,0]) )) / (1+(X@ (np.exp(data_np@beta[k,1:]+beta[k,0]))))) for k in range(num_groups))
  # pdb.set_trace()
  m2.addConstrs(Z[k]+ gp.quicksum([V[j,k]*Y[(j,k)] for j in range(num_products)])== gp.quicksum([price[j]*V[j,k]*X[j] for j in range(num_products)]) for k in range(num_groups))
  
  m2.addConstrs(0<=Y[(j,k)] for j in range(num_products) for k in range(num_groups))
  m2.addConstrs(Y[(j,k)]<=M*X[j] for j in range(num_products) for k in range(num_groups))
  m2.addConstrs(-M*(1-X[j])+Z[k]<=Y[(j,k)] for j in range(num_products) for k in range(num_groups))
  m2.addConstrs(Y[(j,k)]<=Z[k] for j in range(num_products) for k in range(num_groups))
  m2.update()
  return m2,X


data = df1
data_unnormalized = np.array(data)
data_np = np.array(normalization(data,normalization_colomns,stat_dict))
price = data_np[:,-2]
M = price.max()
m_data,X = get_model(data_np,price,M,num_groups = len(theta))
m_data.optimize()
# pdb.set_trace()
print("**********P5,data1,IP**********")
displaye_item = []
for facility in X.keys():
    if (abs(X[facility].x) > 1e-6):
        print(f"displaye #{facility} item.")
        displaye_item.append(int(facility))

price = data_unnormalized[:,-2]
normalized_data = data_np
canidates = displaye_item
print("Suppose unknown the customers type, the displayed canidates are:", set(canidates))
r = return_Revenue_mix(beta,theta,canidates,price,normalized_data)
print("Revenue:",r)
canidates = return_canidate(early_beta,data_np)
print("Suppose known the customers is Type1, the displayed canidates are:", set(canidates))
r = return_Revenue(early_beta,canidates,price,data_np)
print("Revenue:",r)
canidates = return_canidate(late_beta,data_np)
print("Suppose known the customers is Type2, the displayed canidates are:", set(canidates))
r = return_Revenue(late_beta,canidates,price,data_np)
print("Revenue:",r)


data = df2
data_unnormalized = np.array(data)
data_np = np.array(normalization(data,normalization_colomns,stat_dict))
price = data_np[:,-2]
M = price.max()
m_data,X = get_model(data_np,price,M,num_groups = len(theta))
m_data.optimize()
# pdb.set_trace()
print("**********P5,data2,IP**********")
displaye_item = []
for facility in X.keys():
    if (abs(X[facility].x) > 1e-6):
        print(f"displaye #{facility} item.")
        displaye_item.append(int(facility))

price = data_unnormalized[:,-2]
normalized_data = data_np
canidates = displaye_item
print("Suppose unknown the customers type, the displayed canidates are:", set(canidates))
r = return_Revenue_mix(beta,theta,canidates,price,normalized_data)
print("Revenue:",r)
canidates = return_canidate(early_beta,data_np)
print("Suppose known the customers is Type1, the displayed canidates are:", set(canidates))
r = return_Revenue(early_beta,canidates,price,data_np)
print("Revenue:",r)
canidates = return_canidate(late_beta,data_np)
print("Suppose known the customers is Type2, the displayed canidates are:", set(canidates))
r = return_Revenue(late_beta,canidates,price,data_np)
print("Revenue:",r)

data = df3
data_unnormalized = np.array(data)
data_np = np.array(normalization(data,normalization_colomns,stat_dict))
price = data_np[:,-2]
M = price.max()
m_data,X = get_model(data_np,price,M,num_groups = len(theta))
m_data.optimize()
# pdb.set_trace()
print("**********P5,data3,IP**********")
displaye_item = []
for facility in X.keys():
    if (abs(X[facility].x) > 1e-6):
        print(f"displaye #{facility} item.")
        displaye_item.append(int(facility))

price = data_unnormalized[:,-2]
normalized_data = data_np
canidates = displaye_item
print("Suppose unknown the customers type, the displayed canidates are:", set(canidates))
r = return_Revenue_mix(beta,theta,canidates,price,normalized_data)
print("Revenue:",r)
canidates = return_canidate(early_beta,data_np)
print("Suppose known the customers is Type1, the displayed canidates are:", set(canidates))
r = return_Revenue(early_beta,canidates,price,data_np)
print("Revenue:",r)
canidates = return_canidate(late_beta,data_np)
print("Suppose known the customers is Type2, the displayed canidates are:", set(canidates))
r = return_Revenue(late_beta,canidates,price,data_np)
print("Revenue:",r)


data = df4
data_unnormalized = np.array(data)
data_np = np.array(normalization(data,normalization_colomns,stat_dict))
price = data_np[:,-2]
M = price.max()
m_data,X = get_model(data_np,price,M,num_groups = len(theta))
m_data.optimize()
# pdb.set_trace()
print("**********P5,data4,IP**********")
displaye_item = []
for facility in X.keys():
    if (abs(X[facility].x) > 1e-6):
        print(f"displaye #{facility} item.")
        displaye_item.append(int(facility))

price = data_unnormalized[:,-2]
normalized_data = data_np
canidates = displaye_item
print("Suppose unknown the customers type, the displayed canidates are:", set(canidates))
r = return_Revenue_mix(beta,theta,canidates,price,normalized_data)
print("Revenue:",r)
canidates = return_canidate(early_beta,data_np)
print("Suppose known the customers is Type1, the displayed canidates are:", set(canidates))
r = return_Revenue(early_beta,canidates,price,data_np)
print("Revenue:",r)
canidates = return_canidate(late_beta,data_np)
print("Suppose known the customers is Type2, the displayed canidates are:", set(canidates))
r = return_Revenue(late_beta,canidates,price,data_np)
print("Revenue:",r)




late_beta = np.array([-1.53953067,  0.46527555,  0.09313614 , 0.11243684, 0.08595676,   0.02554587,   -0.03388866,   -1.6961064,0.19450294])
late_theta = 0.45690687096001914
early_beta = np.array([-1.9180075,  0.37792879,  0.12591464,  0.09239193, -0.02085044,  0.05785425,  -0.09397025,  -1.07208896,0.13451358])
early_theta = 0.5430931290399809

print("late_beta norm: ",np.linalg.norm(late_beta))
print("early_beta norm: ",np.linalg.norm(early_beta))