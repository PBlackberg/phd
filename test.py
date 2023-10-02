d = {'a':False, 'b':False, 'c':True}

first_true_key = next((key for key, value in d.items() if value), None)

print(first_true_key)