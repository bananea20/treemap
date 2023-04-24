import pandas as pd
import matplotlib.pyplot as plt
import squarify_modify  as  squarify  
import random



def generate_group_list(n):
    group_list = []
    for i in range(n):
        group_list.append("group " + chr(ord('A') + i))
    return group_list

# 用法示例


def generate_weight_list(n, max_weight):
    weight_list = []
    for _ in range(n):
        weight_list.append(random.randint(1, max_weight))
    return weight_list

# 用法示例
# num_weights = 10
# max_weight = 10
# result = generate_weight_list(num_weights, max_weight)
# print(result)




from PIL import Image, ImageDraw

def round_corner_image(image, radius):
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle(
        [0, 0, image.size[0], image.size[1]],
        radius=radius,
        fill=255,
        outline=255,
    )
    rounded_image = Image.composite(image, Image.new("RGBA", image.size), mask)
    return rounded_image

# rounded_img = round_corner_image(img, radius)


# def round_corner_image(image, radius):
#     # 创建一个空白的圆角矩形遮罩
#     mask = Image.new("L", image.size, 0)
#     draw = ImageDraw.Draw(mask)
#     draw.rectangle(
#         [0, 0, image.size[0], image.size[1]],
#         fill=255,
#         outline=255,
#         width=0
#     )

#     # 在遮罩上绘制圆角矩形
#     draw.rounded_rectangle(
#         [0, 0, image.size[0], image.size[1]],
#         radius=radius,
#         fill=0,
#         outline=255
#     )

#     # 将遮罩应用到图片上
#     rounded_image = Image.composite(image, Image.new("RGBA", image.size, (0, 0, 0, 0)), mask)
#     return rounded_image


# 使用 PIL 将图象处理为圆角矩形
# Success
 
from PIL import Image, ImageDraw
 
# radii=10
# img = Image.open('flag.jpg')	
 
# 矩形图像转为圆角矩形
def circle_corner(img, radii):
	# 画圆（用于分离4个角）
 
	circle = Image.new('L', (radii * 2, radii * 2), 0)  # 创建黑色方形
 
	# circle.save('1.jpg','JPEG',qulity=100)
	draw = ImageDraw.Draw(circle)
	draw.ellipse((0, 0, radii * 2, radii * 2), fill=255)  # 黑色方形内切白色圆形
	# circle.save('2.jpg','JPEG',qulity=100)
 
	# 原图转为带有alpha通道（表示透明程度）
	img = img.convert("RGBA")
	w, h = img.size
 
	# 画4个角（将整圆分离为4个部分）
	alpha = Image.new('L', img.size, 255)	#与img同大小的白色矩形，L 表示黑白图
	# alpha.save('3.jpg','JPEG',qulity=100)
	alpha.paste(circle.crop((0, 0, radii, radii)), (0, 0))  # 左上角
	alpha.paste(circle.crop((radii, 0, radii * 2, radii)), (w - radii, 0))  # 右上角
	alpha.paste(circle.crop((radii, radii, radii * 2, radii * 2)), (w - radii, h - radii))  # 右下角
	alpha.paste(circle.crop((0, radii, radii, radii * 2)), (0, h - radii))  # 左下角
	# alpha.save('4.jpg','JPEG',qulity=100)
 
	img.putalpha(alpha)		# 白色区域透明可见，黑色区域不可见

	return img
 