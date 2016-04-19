import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set()


csv_header = ['sum', 'n', 'time']

df_py = pd.read_csv(
    'py.csv', header=None, names=csv_header, sep=' '
)

df_py['method'] = 'numpy'


df_pyopt = pd.read_csv(
    'py_opt.csv', header=None, names=csv_header, sep=' '
)

df_pyopt['method'] = 'opt_einsum'

df_julia = pd.read_csv(
    'julia.csv', header=None, names=csv_header, sep=' '
)

df_julia['method'] = 'julia'

frames = [df_py, df_pyopt, df_julia]

df = pd.concat(frames)


# Initialize a grid of plots with an Axes for each walk
grid = sns.FacetGrid(
    df,
    col="sum",
    hue="method",
    legend_out=False,
    sharey=False,
    col_wrap=3,
    size=3.5
)

# Draw a line plot to show the trajectory of each random walk
grid.map(plt.plot, "n", "time").add_legend()


plt.show()
