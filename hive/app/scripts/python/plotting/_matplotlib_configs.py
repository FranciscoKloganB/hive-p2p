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
    size='xx-large'
)

fp_supertitle = fp_title

fp_subtitle = FontProperties(
    family='sans-serif',
    size='large',
    weight='roman'
)

fp_axis_labels = FontProperties(
    family='sans-serif',
    style='italic',
    weight='semibold',
    size='xx-large'
)

outlyer_shape = {
    # 'markerfacecolor': 'g',
    'marker': 'D'
}
