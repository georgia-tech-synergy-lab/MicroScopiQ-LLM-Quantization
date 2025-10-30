import matplotlib.pyplot as plt

# Sample quartile values for three pairs of box plots
data = [
    {
        'box1': {'whislo': 0.1, 'q1': 1.0, 'med': 2.5, 'q3': 4.0, 'whishi': 6.0, 'fliers': []},
        'box2': {'whislo': 0.2, 'q1': 1.2, 'med': 2.7, 'q3': 4.2, 'whishi': 6.2, 'fliers': []},
        'label': 'Label 1'
    },
    {
        'box1': {'whislo': 0.15, 'q1': 1.1, 'med': 2.6, 'q3': 4.1, 'whishi': 6.1, 'fliers': []},
        'box2': {'whislo': 0.25, 'q1': 1.3, 'med': 2.8, 'q3': 4.3, 'whishi': 6.3, 'fliers': []},
        'label': 'Label 2'
    },
    {
        'box1': {'whislo': 0.2, 'q1': 1.2, 'med': 2.7, 'q3': 4.2, 'whishi': 6.2, 'fliers': []},
        'box2': {'whislo': 0.3, 'q1': 1.4, 'med': 2.9, 'q3': 4.4, 'whishi': 6.4, 'fliers': []},
        'label': 'Label 3'
    }
]

# Creating the figure and axis
fig, ax = plt.subplots()

# Position counter
pos = 1
for item in data:
    # Plot the first box plot
    bplot1 = ax.bxp([item['box1']], positions=[pos - 0.1], patch_artist=True, showfliers=False)
    # Plot the second box plot
    bplot2 = ax.bxp([item['box2']], positions=[pos + 0.1], patch_artist=True, showfliers=False)

    # Customizing the first box plot
    for box in bplot1['boxes']:
        box.set_facecolor('lightblue')
        box.set_edgecolor('black')

    for whisker in bplot1['whiskers']:
        whisker.set_color('black')
        whisker.set_linestyle('-')

    for cap in bplot1['caps']:
        cap.set_color('black')

    for median in bplot1['medians']:
        median.set_color('black')

    # Customizing the second box plot with dotted lines
    for box in bplot2['boxes']:
        box.set_facecolor('lightgreen')
        box.set_edgecolor('black')
        box.set_linestyle('--')  # Dotted line style for the box

    for whisker in bplot2['whiskers']:
        whisker.set_color('black')
        whisker.set_linestyle('--')  # Dotted line style

    for cap in bplot2['caps']:
        cap.set_color('black')
        cap.set_linestyle('--')  # Dotted line style

    for median in bplot2['medians']:
        median.set_color('black')
        median.set_linestyle('--')  # Dotted line style

    pos += 1

# Setting the x-tick labels
ax.set_xticks(range(1, pos))
ax.set_xticklabels([item['label'] for item in data])

# Adding a legend at the top
plt.legend([bplot1["boxes"][0], bplot2["boxes"][0]], ['Box Plot 1', 'Box Plot 2'], loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2)

ax.set_ylabel('Values')

plt.show()


plt.savefig("box_plot.pdf")