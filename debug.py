import random
x = 0
max = 0
for i in range(100): 
    rand = random.random()
    if rand >= 0.5:
        x = x + 1
    else: 
        x = 0
    if x > max: 
        max = x
print(max) 

