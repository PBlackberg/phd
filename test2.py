switch = {'a': False, 'b': True, 'c': False, 'd': True}

# print(next((k for k, v in switch.items() if v), None))
# print(next((k for k, v in (key for key in switch.items() if key[1]) if v), None))




keys = [k for k, v in switch.items() if v]  # This will create a list of keys where value is True
print(keys[0])
print(keys[1])



