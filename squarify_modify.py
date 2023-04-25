# Squarified Treemap Layout
# Implements algorithm from Bruls, Huizing, van Wijk, "Squarified Treemaps"
#   (but not using their pseudocode)


# INTERNAL FUNCTIONS not meant to be used by the user

# 函数的目的是在一个矩形周围添加一个像素宽度的边框，使矩形更加突出
# def pad_rectangle(rect):
#     if rect["dx"] > 2:
#         rect["x"] += 20
#         rect["dx"] -= 0
#     if rect["dy"] > 2:
#         rect["y"] += 20
#         rect["dy"] -= 0
        
# def pad_rectangle(rect):
#     if rect["dx"] > 2:
#         rect["x"] += 1
#         rect["dx"] -= 2
#     if rect["dy"] > 2:
#         rect["y"] += 1
#         rect["dy"] -= 2

def pad_rectangle(rect, n):
    rect["x"] += n
    rect["dx"] -= 2*n
    rect["y"] += n
    rect["dy"] -= 2*n



# 
def layoutrow(sizes, x, y, dx, dy):
    # generate rects for each size in sizes
    # dx >= dy
    # they will fill up height dy, and width will be determined by their area
    # sizes should be pre-normalized wrt dx * dy (i.e., they should be same units)
    # 参数sizes是一个列表，其中包含每个子元素的大小。
        # x和y是布局中左上角的坐标，dx和dy则代表整个布局的宽和高。
        # 在这个函数中，dx要大于或等于dy
    # 该函数首先计算了所有子元素的总面积covered_area，并
        # 基于此计算出每个子元素的宽度width。
        # 然后，函数逐个遍历sizes中的元素，为每个子元素生成一个矩形，
        # 并将其添加到一个矩形列表rects中。

    #每个矩形都有四个属性：x、y、dx和dy。
        # 其中，x和y代表该矩形的左上角坐标，
        # dx代表该矩形的宽度，而dy代表该矩形的高度。

    #最后，函数返回矩形列表rects。
    covered_area = sum(sizes)
    width = covered_area / dy
    rects = []
    for size in sizes:
        rects.append({"x": x, "y": y, "dx": width, "dy": size / width})
        y += size / width
    return rects


def layoutcol(sizes, x, y, dx, dy):
    # generate rects for each size in sizes
    # dx < dy
    # they will fill up width dx, and height will be determined by their area
    # sizes should be pre-normalized wrt dx * dy (i.e., they should be same units)
    covered_area = sum(sizes)
    height = covered_area / dx
    rects = []
    for size in sizes:
        rects.append({"x": x, "y": y, "dx": size / height, "dy": height})
        x += size / height
    return rects


def layout(sizes, x, y, dx, dy):
    return (
        layoutrow(sizes, x, y, dx, dy) if dx >= dy else layoutcol(sizes, x, y, dx, dy)
    )
    # 该函数通过比较dx和dy的大小来确定是使用行布局函数layoutrow还是列布局函数layoutcol，
    # 最终返回一个由布局函数生成的矩形列表。


def leftoverrow(sizes, x, y, dx, dy):
    # compute remaining area when dx >= dy
    covered_area = sum(sizes)
    width = covered_area / dy
    leftover_x = x + width
    leftover_y = y
    leftover_dx = dx - width
    leftover_dy = dy
    return (leftover_x, leftover_y, leftover_dx, leftover_dy)
# 最终，函数返回一个包含剩余空间左上角坐标和宽度高度的元组
#   (leftover_x, leftover_y, leftover_dx, leftover_dy)

def leftovercol(sizes, x, y, dx, dy):
    # compute remaining area when dx >= dy
    covered_area = sum(sizes)
    height = covered_area / dx
    leftover_x = x
    leftover_y = y + height
    leftover_dx = dx
    leftover_dy = dy - height
    return (leftover_x, leftover_y, leftover_dx, leftover_dy)


def leftover(sizes, x, y, dx, dy):
    return (
        leftoverrow(sizes, x, y, dx, dy)
        if dx >= dy
        else leftovercol(sizes, x, y, dx, dy)
    )


def worst_ratio(sizes, x, y, dx, dy):
    return max(
        [
            max(rect["dx"] / rect["dy"], rect["dy"] / rect["dx"])
            for rect in layout(sizes, x, y, dx, dy)
        ]
    )


# PUBLIC API


def squarify(sizes, x, y, dx, dy):
    """Compute treemap rectangles.

    Given a set of values, computes a treemap layout in the specified geometry
    using an algorithm based on Bruls, Huizing, van Wijk, "Squarified Treemaps".
    See README for example usage.

    Parameters
    ----------
    sizes : list-like of numeric values
        The set of values to compute a treemap for. `sizes` must be positive
        values sorted in descending order and they should be normalized to the
        total area (i.e., `dx * dy == sum(sizes)`)
    x, y : numeric
        The coordinates of the "origin".
    dx, dy : numeric
        The full width (`dx`) and height (`dy`) of the treemap.

    Returns
    -------
    list[dict]
        Each dict in the returned list represents a single rectangle in the
        treemap. The order corresponds to the input order.
    """
    sizes = list(map(float, sizes))

    if len(sizes) == 0:
        return []

    if len(sizes) == 1:
        return layout(sizes, x, y, dx, dy)

    # figure out where 'split' should be
    i = 1
    while i < len(sizes) and worst_ratio(sizes[:i], x, y, dx, dy) >= worst_ratio(
        sizes[: (i + 1)], x, y, dx, dy
    ):
        i += 1
    current = sizes[:i]
    remaining = sizes[i:]

    (leftover_x, leftover_y, leftover_dx, leftover_dy) = leftover(current, x, y, dx, dy)
    return layout(current, x, y, dx, dy) + squarify(
        remaining, leftover_x, leftover_y, leftover_dx, leftover_dy
    )
    


def padded_squarify(sizes, x, y, dx, dy,gap_size):
    """Compute padded treemap rectangles.

    See `squarify` docstring for details. The only difference is that the
    returned rectangles have been "padded" to allow for a visible border.
    """
    rects = squarify(sizes, x, y, dx, dy)
    for rect in rects:
        # print("before: ",rect)
        pad_rectangle(rect,gap_size)  
        # print("after: ",rect)
    return rects
'''

padded_squarify函数是用于计算带有内边距的树状图矩形的函数。
它是基于squarify函数的，唯一的区别在于返回的矩形被"填充"了一个可见的
边框，使得树状图更加美观。
padded_squarify函数的输入参数和输出格式与squarify函数相同。

'''


def normalize_sizes(sizes, dx, dy):
    """Normalize list of values.

    Normalizes a list of numeric values so that `sum(sizes) == dx * dy`.

    Parameters
    ----------
    sizes : list-like of numeric values
        Input list of numeric values to normalize.
    dx, dy : numeric
        The dimensions of the full rectangle to normalize total values to.

    Returns
    -------
    list[numeric]
        The normalized values.
    """
    total_size = sum(sizes)
    total_area = dx * dy
    sizes = map(float, sizes)
    sizes = map(lambda size: size * total_area / total_size, sizes)
    return list(sizes)


def plot(
    sizes,
    norm_x=1000,
    norm_y=1000,
    rectangle_size=False,
    color=None,
    label=None,
    value=None,
    ax=None,
    pad=False,
    bar_kwargs=None,
    text_kwargs=None,
    images_paths=None,
    **kwargs
):
    """Plotting with Matplotlib.

    Parameters
    ----------
    sizes
        input for squarify
    norm_x, norm_y
        x and y values for normalization
    color
        color string or list-like (see Matplotlib documentation for details)
    label
        list-like used as label text
    value
        list-like used as value text (in most cases identical with sizes argument)
    ax
        Matplotlib Axes instance
    pad
        draw rectangles with a small gap between them
    bar_kwargs : dict
        keyword arguments passed to matplotlib.Axes.bar
    text_kwargs : dict
        keyword arguments passed to matplotlib.Axes.text
    **kwargs
        Any additional kwargs are merged into `bar_kwargs`. Explicitly provided
        kwargs here will take precedence.

    Returns
    -------
    matplotlib.axes.Axes
        Matplotlib Axes
    """
    
    
    

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, FancyBboxPatch
    import matplotlib.pyplot as plt
    import matplotlib.transforms as transforms
    from PIL import Image, ImageOps
    import numpy as np
    from function import round_corner_image,circle_corner

    if ax is None:
        ax = plt.gca()

    if color is None:
        import matplotlib.cm
        import random

        cmap = matplotlib.cm.get_cmap()
        color = [cmap(random.random()) for i in range(len(sizes))]

    if bar_kwargs is None:
        bar_kwargs = {}
    if text_kwargs is None:
        text_kwargs = {}
    if len(kwargs) > 0:
        bar_kwargs.update(kwargs)

    normed = normalize_sizes(sizes, norm_x, norm_y)

    if pad:
        rects = padded_squarify(normed, 0, 0, norm_x, norm_y,pad)
    else:
        rects = squarify(normed, 0, 0, norm_x, norm_y)

    x = [rect["x"] for rect in rects]
    y = [rect["y"] for rect in rects]
    dx = [rect["dx"] for rect in rects]
    dy = [rect["dy"] for rect in rects]

    if label is None:
        label = []
    if value is None:
        value = []
        
    # print (rects,color, label)    
    # for rect, color,image_path in zip(rects, color,images_paths):
    #     # if is_rectangle:
        
    #         img = Image.open(image_path).convert("RGBA")
    #         img = img.resize((int(rect["dx"]), int(rect["dy"])), Image.BILINEAR)  # 缩放图片至矩形大小
    #         # img.show()
    #         image_array = np.array(img)
            
    #         r = FancyBboxPatch(
    #             (rect["x"], rect["y"]),
    #             rect["dx"],
    #             rect["dy"],
    #             boxstyle=f"round,pad=0,rounding_size={rectangle_size}",
    #             fc=color,
    #             ec="white",
    #             **bar_kwargs
    #         )
    #         ax.add_patch(r)
    #         ax.imshow(
    #             image_array, 
    #             extent=[
    #                 rect["x"], 
    #                 rect["x"] +rect["dx"], 
    #                 rect["y"], 
    #                 rect["y"] + rect["dy"]
    #             ], 
    #             aspect='auto', 
    #             zorder=-1)

                        
                        
    for rect, color, image_path in zip(rects, color, images_paths):
        img = Image.open(image_path).convert("RGBA")
        img = img.resize((int(rect["dx"]), int(rect["dy"])), Image.BILINEAR)
        img = round_corner_image(img, rectangle_size)  # 使用圆角函数
        # img = circle_corner(img, rectangle_size)
        img.show()
        image_array = np.array(img)

        # r = FancyBboxPatch(
        #     (rect["x"], rect["y"]),
        #     rect["dx"],
        #     rect["dy"],
        #     boxstyle=f"round,pad=0,rounding_size={rectangle_size}",
        #     fc='white',
        #     ec="white",
        #     **bar_kwargs
        # )
        # ax.add_patch(r)
        
        ax.imshow(
            image_array,
            extent=[
                rect["x"],
                rect["x"] + rect["dx"],
                rect["y"],
                rect["y"] + rect["dy"],
            ],
            aspect="auto",
            zorder=-1,
        )

            
    # ax.bar(
    #     x, dy, width=dx, bottom=y, color=color, label=label, align="edge", **bar_kwargs
    # )
    


    if not value is None:
        va = "center" if label is None else "top"
        for v, r in zip(value, rects):
            x, y, dx, dy = r["x"], r["y"], r["dx"], r["dy"]
            ax.text(x + dx / 2, y + dy / 2, v, va=va, ha="center", **text_kwargs)


    if not label is None:
        va = "center" if value is None else "bottom"
        # print(label)
        for l, r in zip(label, rects):
            # print (l)
            x, y, dx, dy = r["x"], r["y"], r["dx"], r["dy"]
            ax.text(x + dx / 2, y + dy / 2, l, va=va, ha="center", **text_kwargs)

    ax.set_xlim(0, norm_x)
    ax.set_ylim(0, norm_y)

    return ax





