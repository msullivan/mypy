import pickle
def crunch(d):
    d2 = {}
    for (k, _), v in d.items():
        d2[k] = d2.get(k, 0) + v
    return d2
def crunch2(d):
    d2 = {}
    for (_, k), v in d.items():
        d2[k] = d2.get(k, 0) + v
    return d2

d = pickle.load(open('lol.pickle', 'rb'))
d2 = crunch(d)
d3 = crunch(d)

total=sum(d.values())
for x, y in sorted(d.items(), key=lambda x: x[1]):
    print("{:8}  ({:2.2f}): {}".format(y, 100*y/total, x))
