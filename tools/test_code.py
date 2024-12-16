
# points = []
# for i in range(240):
#     if i % 5 == 0:
#         points.append(i)
# points = [item + 1 for item in points]
# first_order_mempoints = points[::4]
# second_order_mempoints = points[1::4]
# third_order_mempoints = points[2::4]
# review_points = points[3::4]
# memory_points = sorted(first_order_mempoints + second_order_mempoints + third_order_mempoints)
#
# print(review_points)
# print(first_order_mempoints)
# print(second_order_mempoints)
# print(third_order_mempoints)
# print(memory_points)

# previous_features = [[0 for col in range(3)] for row in
#                              range(3)]
#
# previous_features[0][0] = 1
# previous_features[0][1] = 2
# previous_features[0][2] = 3
# previous_features[1][0] = 4
# previous_features[1][1] = 5
# previous_features[1][2] = 6
# previous_features[2][0] = 7
# previous_features[2][1] = 8
# previous_features[2][2] = 9
# print(previous_features)
#
# previous_features[0][0:3] = [1,2,3]
#
# print(previous_features)

l1 = [1,2,3]
l2 = [4,5,6]
l3 = [7,8,9]
previous_features = []

previous_features.append(l1)
previous_features.append(l2)
previous_features.append(l3)
print(previous_features)