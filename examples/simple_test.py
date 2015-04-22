from annoy import AnnoyIndex

a = AnnoyIndex(3)
a.add_item(0, [1, 0, 0], 1)
a.add_item(1, [0, 1, 0], 1)
a.add_item(2, [0, 0, 1], 1)
a.build(-1)

print(a.get_nns_by_item(1, 100, [2], 10))
print(a.get_nns_by_vector([1.0, 0.5, 0.5], 100, [], 1000))
