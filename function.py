import pandas as pd
import matplotlib.pyplot as plt


def draw_treemap(weight_list, label_list):

    df = pd.DataFrame({'weight':weight_list, 'group':label_list })

    # plot it
    
    squarify.plot(sizes=df['weight'], label=df['group'], alpha=.8,rectangle_size=rectangle_size,pad=Gap_size,value=value)
    plt.axis('off')
    plt.show()
    