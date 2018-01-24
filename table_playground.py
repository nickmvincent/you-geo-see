
import matplotlib.pyplot as plt


fig, axes = plt.subplots()


the_labels = plt.table(cellText=[[0, 1, 2], [3, 4, 5]], loc='center')
for key, cell in the_labels.get_celld().items():
    cell.set_linestyle('--')
    cell.set_linewidth(4)
    cell.set_edgecolor('lightgray')
axes.add_table(the_labels)
plt.show()