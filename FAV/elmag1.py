import numpy as np


#print(np.arctan2(np.sqrt(3),1)*180/np.pi)

def xz_to_r_theta(x, z):
    r = np.sqrt(x**2 + (z-d/2)**2)
    theta = abs(np.arctan2(x, (z-d/2))*180/np.pi)
    return r, theta

def r_theta_to_xz(r, theta):
    x = r * np.sin(theta*np.pi/180)
    z = r * np.cos(theta*np.pi/180) + d/2
    return x, z

def Ecoulomb(q, dist):
    return k*q/(dist**2)

def phi(q, dist):
    return k*q/dist

def psi(dist):
    return k/dist

def psi(x, z, qz):
    # 1. Reálný náboj
    r0 = np.sqrt(x**2 + (z - qz)**2)
    # 2. Zrcadlení přes dolní desku (z=0)
    r1 = np.sqrt(x**2 + (z + qz)**2)
    # 3. Zrcadlení reálného náboje přes horní desku (z=h)
    r2 = np.sqrt(x**2 + (z - (2*h - qz))**2)
    # 4. Zrcadlení prvního obrazu přes horní desku
    r3 = np.sqrt(x**2 + (z - (2*h + qz))**2)
    # 5. Zrcadlení druhého obrazu přes dolní desku
    r4 = np.sqrt(x**2 + (z - (-2*h + qz))**2)
    
    # Kombinace potenciálů (střídání znamének udržuje desky na konstantním potenciálu)
    return k * (1/r0 - 1/r1 - 1/r2 + 1/r3 + 1/r4)

epsilon =  8.854187817e-12 # F/m
epsilon2 = epsilon * 1e9 * 1e-2 # nC / V / cm
fourpiepsilon = 4*np.pi*epsilon2
k = 1/fourpiepsilon


h = 11.5
d = 5
U = 3000
Unanovolt = U*1e9

def Ehom(z):
    return U/h * z

def dist(x1, z1, x2, z2):
    return np.sqrt((x1-x2)**2 + (z1-z2)**2)

qz1 = d/2
delta = 0.5 # cm
qz2 = d/2 + delta
qz3 = d/2 - delta
print(qz1, qz2, qz3)

X1, Z1 = r_theta_to_xz(r=d/2, theta=0)
print(X1, Z1)
b1 = -Ehom(Z1)
X2, Z2 = r_theta_to_xz(r=d/2, theta=60)
print(X2/d*2, Z2)
b2 = -Ehom(Z2)
X3, Z3 = r_theta_to_xz(r=d/2, theta=120)
print(X3/d*2, Z3)
b3 = -Ehom(Z3)
b = [b1, b2, b3]
a11, a12, a13, a21, a22, a23, a31, a32, a33 = 1, 2, 3, 4, 5, 6, 7, 8, 9
# a = np.array([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])
# for i in range(len(b)):
#     print(float(b[i]))
# for i in range(len(a)):
#     for j in range(len(a[i])):
#         print(float(a[i][j]), end=' ')
#     print()
# a11 = psi(dist(0, qz1, X1, Z1))
# a12 = psi(dist(0, qz2, X1, Z1))
# a13 = psi(dist(0, qz3, X1, Z1))

# a21 = psi(dist(0, qz1, X2, Z2))
# a22 = psi(dist(0, qz2, X2, Z2))
# a23 = psi(dist(0, qz3, X2, Z2))

# a31 = psi(dist(0, qz1, X3, Z3))
# a32 = psi(dist(0, qz2, X3, Z3))
# a33 = psi(dist(0, qz3, X3, Z3))
print('psi(0, 0, d/2):', psi(0, d, d/2))

a11 = psi(X1, Z1, qz1)
a12 = psi(X1, Z1, qz2)
a13 = psi(X1, Z1, qz3)

a21 = psi(X2, Z2, qz1)
a22 = psi(X2, Z2, qz2)
a23 = psi(X2, Z2, qz3)

a31 = psi(X3, Z3, qz1)
a32 = psi(X3, Z3, qz2)
a33 = psi(X3, Z3, qz3)

a = np.array([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])
#print(b)
for i in range(len(b)):
    print(float(b[i]))
for i in range(len(a)):
    for j in range(len(a[i])):
        print(float(a[i][j]), end=' ')
    print()

charges = np.linalg.solve(a, b)
print(charges)

def potential(x, z):
    phi_total = 0
    phi_hom = Ehom(z)
    for qz, q in zip([qz1, qz2, qz3], charges):
        r_real = np.sqrt(x**2 + (z - qz)**2)
        r_mirror = np.sqrt(x**2 + (z + qz)**2)
        phi_total += k * q * (1/r_real - 1/r_mirror)
    return phi_total + phi_hom

print(potential(0, 0), potential(0, d), potential(X2, Z2))
