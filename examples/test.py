def convert(base,n):
    temp = n
    octal = 0
    ctr = 0

    while temp > 0:
        octal += (temp%base) * (10**ctr)
        temp = temp//base
        ctr += 1

    return octal


base = 4

for i in range(base**2):
    x = convert(base,i)
 
    print([int(a) for a in str(x)])
