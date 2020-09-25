import pandas as pd
import plotly.express as px
from Visualization.functions import *

ptf = 'CORE 6'
what = 'sub_asset_class'
df_raw = pd.read_csv('ptf_hist.csv').sort_values(what)
df_raw = df_raw[df_raw.name_id == ptf]
df = df_raw.pivot_table(['weight'], index=['ref_date', what], aggfunc='sum').reset_index()
df.loc[:, 'ref_date'] = [pd.to_datetime(i) for i in df.ref_date]
df = df[df.ref_date > pd.to_datetime('2015-01-01')]
fig = px.line(df,
              y="weight",
              x="ref_date",
              #animation_frame="ref_date",
              range_x=[min(df.ref_date), max(df.ref_date)],
              color=what,
              hover_name=what)

# improve aesthetics (size, grids etc.)


fig.update_xaxes(title_text='Reallocation Date')
fig.update_yaxes(title_text='SAC')
fig.show()
quit()
fig = px.bar(df,
             y=what,
             x="weight",
             animation_frame="ref_date",
             orientation='v',
             range_x=[0, df.weight.max()],
             color=what)

# improve aesthetics (size, grids etc.)

fig.update_layout(width=1000,
                  height=800,
                  xaxis_showgrid=False,
                  yaxis_showgrid=False,
                  paper_bgcolor='rgba(0,0,0,0)',
                  plot_bgcolor='rgba(0,0,0,0)',
                  title_text=ptf + ' Sub Asset Classes',
                  showlegend=True)


fig.update_xaxes(title_text='Reallocation Date')
fig.update_yaxes(title_text='SAC')
fig.show()

quit()
