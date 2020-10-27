import matplotlib.style
import matplotlib.pyplot

import matplotlib as mpl

from matplotlib.font_manager import FontProperties

mpl.style.use('seaborn-deep')

color_palette = matplotlib.pyplot.style.library['seaborn-deep']['axes.prop_cycle'].by_key()['color']

title_pad = 20
labels_pad = 15
legends_pad = 12
ax_alpha = 0.8

fp_title = FontProperties(
    family='sans-serif',
    weight='bold',
    size='xx-large'
)

fp_supertitle = fp_title

fp_subtitle = FontProperties(
    family='sans-serif',
    size='x-large',
    weight='semibold'
)

fp_axis_labels = FontProperties(
    family='sans-serif',
    style='italic',
    weight='semibold',
    size='xx-large'
)

fp_axis_legend = FontProperties(
    family='sans-serif',
    style='italic',
    weight='roman',
    size='x-large'
)

outlyer_shape = {
    # 'markerfacecolor': 'g',
    'marker': 'D'
}
