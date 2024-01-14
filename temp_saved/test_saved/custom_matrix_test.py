import numpy as np
import xarray as xr



# Creating a 3-dimensional numpy array (shape is (layers, rows, columns))
my_3d_array = np.array([
    [                   # layer 0
    [1, 2, 3],              # row 0
    [4, np.nan, 6],         # ...
    [1, 2, 3],
    [1, np.nan, 3],
    [1, 2, 3]
    ],

    [
    [7, 8, 9],          # layer 1
    [10, np.nan, 12],       # row 0
    [1, 2, 3],              # ...
    [1, 2, 3],
    [1, 2, 3]
    ],

    [                   # ...
    [13, 14, 15], 
    [16, 17, 18],
    [1, 2, 3],
    [1, 2, 3],
    [1, 2, 3]
    ],

    [
    [13, 14, 15], 
    [16, 17, 18],
    [1, 2, 3],
    [1, 2, 3],
    [1, 2, 3]
    ]
    ])
# print(my_3d_array)
# print(my_3d_array[0, :, 1])


# 4D data array (groups, layers, rows, columns)
my_4d_array = np.array([
    [                   # group 0
    [                       # layer 0
    [1, 2, 3],                  # row 0
    [4, np.nan, 6],             # row 1
    ],

    [                       # layer 1
    [7, 8, 9],                  # row 0
    [10, 11, 12],               # row 1
    ],
    ],


    [                   # group 1
    [                       # layer 0
    [13, 14, 15],               # row 0
    [16, 17, 18],               # row 1
    ],

    [                       # layer 1
    [19, 20, 21],               # row 0
    [22, 23, 24],               # row 1
    ],
    ],
    ])
# print(my_4d_array)
print(my_4d_array[0,1,1,:])


