import matplotlib.style
import matplotlib as mpl

from matplotlib.font_manager import FontProperties

mpl.style.use('seaborn-deep')

title_pad = 20
labels_pad = 15
legends_pad = 12

fp_title = FontProperties(
    family='sans-serif',
    weight='bold',
    size='x-large'
)

fp_subtitle = FontProperties(
    family='sans-serif',
    size='large',
    weight='demibold'
)

fp_axis_labels = FontProperties(
    family='sans-serif',
    style='italic',
    weight='semibold',
    size='x-large'
)

outlyer_shape = {
    # 'markerfacecolor': 'g',
    'marker': 'D'
}
