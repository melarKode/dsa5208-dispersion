"""Plot recorded plume moments without reconstructing concentration fields."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('review', type=Path)
a = p.parse_args()
d = np.genfromtxt(a.review / 'diagnostics.csv', delimiter=',', names=True)
t = d['time']
plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 11})
fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
ax = axes[0, 0]
ax.plot(d['mean_x'], d['mean_y'], color='#2563eb', alpha=.6)
ax.scatter(d['mean_x'], d['mean_y'], c=t, cmap='viridis', s=65, zorder=3)
for x, y, time in zip(d['mean_x'], d['mean_y'], t):
    ax.annotate(f'{time:g} s', (x, y), xytext=(8, 0), textcoords='offset points', fontsize=9)
ax.set(xlabel='x (m)', ylabel='y (m)', title='Mass-weighted centre: horizontal path', xlim=(50, 250))
ax.text(.02, .02, 'Axes use different scales; points are centres, not plume outlines.', transform=ax.transAxes, fontsize=8)
ax = axes[0, 1]
ax.plot(t, d['mean_z'], 'o-', color='#2563eb')
ax.set(xlabel='Time (s)', ylabel='Mean z coordinate (m)', title='Mean z of pollutant still inside')
ax = axes[1, 0]
for axis, color in zip('xyz', ['#2563eb', '#e07a18', '#15956e']):
    ax.plot(t, np.sqrt(np.maximum(d['var_'+axis], 0)), 'o-', color=color, label=axis)
ax.set(xlabel='Time (s)', ylabel='Standard deviation (m)', title='Spread of pollutant still inside')
ax.legend(title='Direction')
ax = axes[1, 1]
ax.semilogy(t, d['max_concentration'] / d['max_concentration'][0], 'o-', color='#9333ea')
ax.set(xlabel='Time (s)', ylabel='Peak / initial peak (log scale)', title='Peak concentration relative to release')
for ax in axes.flat:
    ax.grid(alpha=.2)
fig.suptitle(f'Urban transport · {t[-1]:g} seconds\nMoments describe pollutant remaining inside the domain', fontsize=16)
fig.savefig(a.review / 'pilot-summary.png', dpi=170)
print(a.review / 'pilot-summary.png')
