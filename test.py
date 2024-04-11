y=  [0,0,1,1,0]
a = (0.2,0.4,0.6,0.8,1)

from math import log10

def prob_loss(y,a):
    vals= []
    for i in range(len(y)):
        x = a[i]

        vals.append(log10(1-x))


        # if y[i] == 1:
        #     vals.append(log10(a[i]))
        # if y[i] == 0:
        #     vals.append(log10(1-(a[i])))
    return sum(vals)

    
prob_loss(y,a)


