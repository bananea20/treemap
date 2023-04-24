import pandas as pd
import matplotlib.pyplot as plt
import squarify_modify  as  squarify  
from function import *
# Create a data frame with fake data



weight_list = [8,3,4,2]
label_list = ["group A", "group B", "group C", "group D"]
value = [1,2,3,4]



# 间隙像素,如果为0则没有间隙
Gap_size=2
# 圆角半径，如果为0则没有圆角
rectangle_size = 5

def draw_treemap(weight_list, label_list):

    df = pd.DataFrame({'weight':weight_list, 'group':label_list })

    # plot it
    
    squarify.plot(sizes=df['weight'], label=df['group'], alpha=.8,rectangle_size=rectangle_size,pad=Gap_size,value=value)
    plt.axis('off')
    plt.show()
    

    
#  img = round_corner_image(img, 10)  # 使用圆角函数