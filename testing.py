# vars_highlight = None #'astring' #['a', 'b']
# print(type(vars_highlight))
# # exit()

# import xarray as xr
# vars_highlight = xr.Dataset(data_vars = {'a': xr.DataArray([1,2,3])})

# if vars_highlight is None:
#     print('executes')

# exit()
# if type(vars_highlight) == list:
#     print('executes')

# if type(vars_highlight) == int:
#     print('executes')

# if type(vars_highlight) == str:
#     print('executes')




# a = ['1', 'a', 'b']
# print('\n'.join(i for i in a))



# import numpy as np
# import matplotlib.pyplot as plt
# a = np.array([1, 2, 3])

# plt.figure()
# plt.plot(a)

# plt.show()



import numpy as np
import matplotlib.colors as mcolors

def generate_distinct_colors(n):
    hsv_colors = [(i / n, 1, 1) for i in range(n)]  # Hue varies, saturation and value are maxed
    rgb_colors = [mcolors.hsv_to_rgb(color) for color in hsv_colors]
    return rgb_colors

# Example: Generate 5 distinct colors
n_colors = 5
colors = generate_distinct_colors(n_colors)
print(colors)