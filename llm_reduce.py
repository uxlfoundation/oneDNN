import re
import random

def reduce(dim):
    MAX, MOD = 1024, 256
    if dim <= MAX: return dim
    rem = dim % MOD
    if rem == 0: return random.randrange(MOD, MAX + 1, MOD)
    return rem + random.randrange(0, MAX, MOD)

random.seed(0)
lines = open('tests/benchdnn/inputs/matmul/option_set_fwks_llm_gpu').readlines()
for l in lines:
    m = re.search(r'(\w+):(\w+)_n"', l)
    s, a, b = m.group(0), m.group(1), m.group(2)
    a_dims = list(map(int, a.split('x')))
    b_dims = list(map(int, b.split('x')))
    m, n = a_dims[-2], b_dims[-1]
    m_red, n_red = reduce(m), reduce(n)
    a_dims[-2] = m_red
    b_dims[-1] = n_red
    a = 'x'.join(map(str, a_dims))
    b = 'x'.join(map(str, b_dims))
    l = l.replace(s, f'{a}:{b}_n"')
    print(l.strip())
