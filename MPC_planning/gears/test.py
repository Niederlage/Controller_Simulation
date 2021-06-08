def get_gcd(a, b):
    if a < b:
        a, b = b, a
    if abs(a) < 1e-10:
        a = 0
        return 0
    if abs(b) < 1e-10:
        b = 0
        return 0

    y = a % b
    if y == 0:
        return b
    else:
        a, b = b, y

    return get_gcd(a, b)


a = 1.6
b = 3.2
c = 6.4
ab = get_gcd(a, b)
bc = get_gcd(b, c)
print(1)
