import io
import numpy as np
import time
from numpy.lib.stride_tricks import as_strided

def loads(b, max_integer_width = 4, encoding = 'utf-8', delimiter = '\t', newline = '\n', comments = '#'):
	assert b[-1] == ord(newline)

	has_names = b[0] == ord(comments)
	head = np.genfromtxt(io.BytesIO(b), max_rows = 2 if has_names else 1, delimiter = delimiter, names = True if has_names else None, dtype = None, encoding = encoding)
	uniform = len(head.dtype.descr) == 1 and not head.dtype.descr[0][0]

	integer_cols = [(i, n) for i, (n, t) in enumerate(head.dtype.descr) if np.issubdtype(np.dtype(t), np.integer)] if not uniform else np.arange(head.shape[-1], dtype = int) if np.issubdtype(head.dtype, np.integer) else np.array([], dtype = int)
	float_cols = [(i, n) for i, (n, t) in enumerate(head.dtype.descr) if np.issubdtype(np.dtype(t), np.floating)] if not uniform else np.arange(head.shape[-1], dtype = int) if np.issubdtype(head.dtype, np.floating) else np.array([], dtype = int)
	integers = np.array(list(zip(*integer_cols))[0]) if not uniform else integer_cols
	floats = np.array(list(zip(*float_cols))[0]) if not uniform else float_cols

	a = np.frombuffer(b, dtype = np.uint8)

	newlines = a == ord(newline)
	if has_names:
		idx = np.flatnonzero(newlines)[0] + 1
		a = a[idx:]
		newlines = newlines[idx:]

	tabs = a == ord(delimiter)
	minus = a == ord(b'-')
	num_rows = newlines.sum()

	a0 = np.zeros((max_integer_width - 1 + len(a), ), dtype = np.uint8)
	np.subtract(a, ord(b'0'), out = a0[max_integer_width - 1:])
	m = as_strided(a0, shape = [max_integer_width, len(a)], strides = a0.strides * 2)[::-1]
	m = m * np.power(10, np.arange(max_integer_width), dtype = np.int32)[:, None]
	for i in range(1, max_integer_width):
		np.add(m[i], m[i - 1], out = m[i])

	breaksi = np.flatnonzero(tabs | newlines)

	if len(integer_cols) > 0:
		widthi = np.diff(np.pad(breaksi, (1, 0), mode = 'constant', constant_values = -1))
		BI = (breaksi.reshape(num_rows, -1) - 1)[:, integers].flatten()
		WI = (widthi.reshape(num_rows, -1) - 2)[:, integers].flatten()
		resi = m[WI, BI].reshape(num_rows, -1)

	if len(floats) > 0:
		points = a == ord(b'.')
		breaksf = np.flatnonzero(points)
		BF = (breaksi.reshape(num_rows, -1) - 1)[:, floats].flatten()
		BF_ = np.vstack([breaksf - 1, BF]).T.flatten()
		BF__ = (breaksi.reshape(num_rows, -1))[:, floats - 1].flatten()
		widthf = BF - breaksf
		WF_ = np.vstack([breaksf - BF__ - 1, widthf]).T.flatten() - 1
		resf = m[WF_, BF_].reshape(num_rows, -1)
		resf = np.ascontiguousarray(resf.astype(np.float32).reshape(-1, 2).T)
		np.multiply(resf[1], np.power(10.0, -widthf, dtype = np.float32), out = resf[1])
		resf = np.add(resf[0], resf[1], out = resf[1]).reshape(num_rows, -1)

	if not uniform:
		integer_cols = {n : j for j, (i, n) in enumerate(integer_cols)}
		float_cols = {n : j for j, (i, n) in enumerate(float_cols)}
		return np.rec.fromarrays([resi[:, integer_cols[n]] if n in integer_cols else resf[:, float_cols[n]] for n in head.dtype.names], names=head.dtype.names)
	else:
		return resi if len(integers) > 0 else resf
		
if __name__ == '__main__':
	# python3 -m cProfile -s tottime tsv.py

	tic = time.time()
	#print(loads(b'# a\tb\tc\td\n1\t22.60\t3\t5.0\n3\t44.8\t8\t9.09\n'))

	#print(tsvparse(b'1\t22\n3\t44\n', integers = True))
	#np.savetxt('test.txt', np.random.randint(0, 10000, size = (100000, 20)),fmt = '%d', delimiter = '\t')
	#np.savetxt('test2.txt', np.random.rand(100000, 20) * 10,fmt = '%.4f', delimiter = '\t')
	
	# baseline
	#res = np.loadtxt(io.StringIO(b.decode('ascii')), dtype = np.float32 if floats else int, delimiter = '\t')

	print(loads(open('test.txt', 'rb').read()).shape)
	#M = loads(open('test.txt', 'rb').read())


	print(time.time() - tic)
