import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
df=pd.read_csv(ROOT/"data/hypothetical_logistics_data.csv")
OUT=ROOT/"visualizations"; OUT.mkdir(exist_ok=True)
print(df.describe())
print("\nMissing values:\n",df.isna().sum())
print("\nCorrelations:\n",df.select_dtypes("number").corr())

plt.hist(df.delivery_time_hours,bins=25); plt.title("Distribution of Delivery Time"); plt.xlabel("Hours"); plt.ylabel("Shipments"); plt.savefig(OUT/"01_delivery_time_distribution.png",dpi=150,bbox_inches="tight"); plt.close()
x=df.groupby("region").delay_hours.mean(); plt.bar(x.index,x.values); plt.title("Average Delay by Region"); plt.ylabel("Hours"); plt.savefig(OUT/"02_average_delay_by_region.png",dpi=150,bbox_inches="tight"); plt.close()
plt.scatter(df.distance_km,df.transportation_cost,alpha=.55); plt.title("Distance vs Transportation Cost"); plt.xlabel("Distance (km)"); plt.ylabel("Cost"); plt.savefig(OUT/"03_distance_vs_cost.png",dpi=150,bbox_inches="tight"); plt.close()
x=df.groupby("transport_mode").transportation_cost.mean(); plt.bar(x.index,x.values); plt.title("Average Transportation Cost by Mode"); plt.savefig(OUT/"04_cost_by_mode.png",dpi=150,bbox_inches="tight"); plt.close()
x=df.groupby("traffic_level").delay_hours.mean(); plt.bar(x.index,x.values); plt.title("Average Delay by Traffic Level"); plt.ylabel("Hours"); plt.savefig(OUT/"05_delay_by_traffic.png",dpi=150,bbox_inches="tight"); plt.close()
c=df.select_dtypes("number").corr(); fig,ax=plt.subplots(figsize=(8,6)); im=ax.imshow(c); ax.set_xticks(range(len(c))); ax.set_xticklabels(c.columns,rotation=35,ha="right"); ax.set_yticks(range(len(c))); ax.set_yticklabels(c.columns)
for i in range(len(c)):
    for j in range(len(c)): ax.text(j,i,f"{c.iloc[i,j]:.2f}",ha="center",va="center")
ax.set_title("Correlation Matrix"); fig.colorbar(im,ax=ax); fig.savefig(OUT/"06_correlation_matrix.png",dpi=150,bbox_inches="tight"); plt.close()
