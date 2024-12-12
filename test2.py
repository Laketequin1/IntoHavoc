abs_pos = [0, 0, 0]

axes = [i for i, x in enumerate(abs_pos) if x > 1]
print(axes)
possible_axes = [0, 1, 2]
removed_axes = [axis for axis in possible_axes if axis not in axes]
print(removed_axes)