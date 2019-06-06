import io
import numpy as np
import time

def tsvparse(b, max_integer_width = 4, integers = [], floats = []):
    assert b[-1] == ord(b'\n') and b[0] != ord(b'#')
    tic = time.time()

    a = np.frombuffer(b, dtype = np.uint8)
    newlines    = a == ord(b'\n')
    tabs        = a == ord(b'\t')
    points      = a == ord(b'.')
    num_rows = newlines.sum()

    breaksi = np.flatnonzero(tabs | newlines)
    breaksf = np.flatnonzero(points)

    ai = (a - ord(b'0')).astype(np.int32)
    m = [ai]
    for p in range(1, max_integer_width):
        m.append(m[-1] + 10 * np.pad(ai[:-p], (p, 0), mode = 'constant'))
    m = np.vstack(m)

    widthi = np.diff(np.pad(breaksi, (1, 0), mode = 'constant', constant_values = -1))
    BI = (breaksi.reshape(num_rows, -1) - 1)[:, integers].flatten()
    WI = (widthi.reshape(num_rows, -1) - 2)[:, integers].flatten()
    resi = m[WI, BI].reshape(num_rows, -1)

    BF = (breaksi.reshape(num_rows, -1) - 1)[:, floats].flatten()
    BF_ = np.vstack([breaksf - 1, BF]).T.flatten()
    BF__ = (breaksi.reshape(num_rows, -1))[:, np.array(floats) - 1].flatten()
    widthf = BF - breaksf
    WF_ = np.vstack([breaksf - BF__ - 1, widthf]).T.flatten() - 1
    resf = m[WF_, BF_].reshape(num_rows, -1)

    resf = np.ascontiguousarray(resf.astype(np.float32).reshape(-1, 2).T)
    np.multiply(resf[1], np.power(10.0, -widthf, dtype = np.float32), out = resf[1])
    resf = np.add(resf[0], resf[1], out = resf[1]).reshape(num_rows, -1)

    print(time.time() - tic)
    return resi, resf
		
def tsvparse2(b, integers = False, floats = False):
    f = io.StringIO(b.decode('ascii'))
    tic = time.time()
    res = np.loadtxt(f, dtype = np.float32 if floats else int, delimiter = '\t')
    print(time.time() - tic)
    return res

if __name__ == '__main__':
    print(tsvparse(b'1\t22.60\t3\t5.0\n3\t44.8\t8\t9.09\n', integers = [0, 2], floats = [1, 3]))

    #print(tsvparse(b'1\t22\n3\t44\n', integers = True))
    #np.savetxt('test.txt', np.random.randint(0, 10000, size = (100000, 20)),fmt = '%d', delimiter = '\t')
    #np.savetxt('test2.txt', np.random.rand(100000, 20) * 10,fmt = '%.4f', delimiter = '\t')

    #print(tsvparse(open('test2.txt', 'rb').read(), floats = True).shape)
