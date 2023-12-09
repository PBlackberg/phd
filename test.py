# Initial setup
example_list = [None] * 10  # A list pre-filled with None
small_list_1 = [1, 2, 3]
small_list_2 = [4, 5]


idx = 0
print(idx)
print(idx + len(small_list_1))
example_list[idx:idx + len(small_list_1)] = small_list_1
idx += len(small_list_1)  # idx is now 3


print(idx)
print(idx + len(small_list_2))
example_list[idx:idx + len(small_list_2)] = small_list_2
idx += len(small_list_2)  # idx is now 5
print(idx)


print(example_list)




# print(da)
# plot_object = da.isel(time=0).plot()
# fig = plot_object.figure
# fig.savefig(f'{os.getcwd()}/test/plot_test/test.png')
# print(da.isel(time=0))