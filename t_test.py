import pandas as pd
from scipy.stats import ttest_ind
import numpy as np


h1 = pd.read_csv("f_h_1.csv")
c1 = pd.read_csv("f_c_1.csv")
h2 = pd.read_csv("f_h_2.csv")
c2 = pd.read_csv("f_c_2.csv")
h3 = pd.read_csv("f_h_3.csv")
c3 = pd.read_csv("f_c_3.csv")
c4 = pd.read_csv("f_c_4.csv")
c5 = pd.read_csv("f_c_5.csv")

data = pd.concat([h1, c1, h2, c2, c4, h3, c4, c5], ignore_index=True)

print(data.shape)

feature_cols = [
    c for c in data.columns
    if c not in ["window_id","row","col","label"]
]

healthy_data = data[data["label"]==0]
crack_data = data[data["label"]==1]

ttest_results=[]

for feature in feature_cols:

    t_stat,p = ttest_ind(
        healthy_data[feature],
        crack_data[feature],
        equal_var=False
    )

    ttest_results.append({
        "Feature":feature,
        "t_value":t_stat,
        "p_value":p
    })

ttest_df = pd.DataFrame(ttest_results)

#print(ttest_df)

ttest_df.to_csv("ttest_results.csv",index=False)


def cohens_d(x,y):

    nx=len(x)
    ny=len(y)

    pooled=np.sqrt(
        (
            (nx-1)*np.var(x,ddof=1)+
            (ny-1)*np.var(y,ddof=1)
        )/(nx+ny-2)
    )

    d=(np.mean(x)-np.mean(y))/pooled

    return d

effect=[]

for feature in feature_cols:

    d=cohens_d(
        healthy_data[feature],
        crack_data[feature]
    )

    effect.append({
        "Feature":feature,
        "Cohens_d":d
    })

effect_df=pd.DataFrame(effect)

effect_df.to_csv(
    "cohens_d.csv",
    index=False
)

#print(effect_df)

statistics = ttest_df.merge(
    effect_df,
    on="Feature"
)

statistics.to_csv(
    "Feature_Statistics.csv",
    index=False
)

#print(statistics)

selected = statistics[
    (statistics["p_value"]<0.05) &
    (statistics["Cohens_d"].abs()>0.5)
]

#print(selected)