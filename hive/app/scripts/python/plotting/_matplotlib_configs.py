import matplotlib.style
import matplotlib as mpl

from matplotlib.font_manager import FontProperties

mpl.style.use('seaborn-deep')

fp_axis_labels = FontProperties()
fp_axis_labels.set_family('sans-serif')
fp_axis_labels.set_size('large')
fp_axis_labels.set_name('Times New Roman')
fp_axis_labels.set_weight('bold')
