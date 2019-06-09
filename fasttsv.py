import numpy as np
from numpy.lib.stride_tricks import as_strided

def loads(b, max_integer_width = None, encoding = 'utf-8', delimiter = '\t', newline = '\n', comments = '#', decimal_point = '.'):
	assert b[-1] == ord(newline)

	has_names = b[0] == ord(comments)
	head = np.genfromtxt(io.BytesIO(b), max_rows = 2 if has_names else 1, delimiter = delimiter, names = True if has_names else None, dtype = None, encoding = encoding)
	uniform = len(head.dtype.descr) == 1 and not head.dtype.descr[0][0]

	integer_cols = [(i, n) for i, (n, t) in enumerate(head.dtype.descr) if np.issubdtype(np.dtype(t), np.integer)] if not uniform else np.arange(head.shape[-1], dtype = int) if np.issubdtype(head.dtype, np.integer) else np.array([], dtype = int)
	float_cols = [(i, n) for i, (n, t) in enumerate(head.dtype.descr) if np.issubdtype(np.dtype(t), np.floating)] if not uniform else np.arange(head.shape[-1], dtype = int) if np.issubdtype(head.dtype, np.floating) else np.array([], dtype = int)
	integers = np.array(list(zip(*integer_cols))[0]) if not uniform else slice(None)
	floats = np.array(list(zip(*float_cols))[0]) if not uniform else slice(None)

	a = np.frombuffer(b, dtype = np.uint8)

	newlines = a == np.uint8(ord(newline))
	if has_names:
		idx = np.flatnonzero(newlines)[0] + 1
		a = a[idx:]
		newlines = newlines[idx:]

	tabs = a == np.uint8(ord(delimiter))
	minus = a == np.uint8(ord(b'-'))
	num_rows = newlines.sum()

	breaksi = np.flatnonzero(np.bitwise_or(tabs, newlines, out = tabs)).reshape(num_rows, -1)
	breaksi_max = breaksi.max()
	for dt in [np.int8, np.int16, np.int32]:
		if breaksi_max < np.iinfo(dt).max:
			breaksi = breaksi.astype(dt, copy = False)
			break

	breaksi -= 1
	widthi = np.diff(breaksi.ravel(), prepend = -2).reshape(num_rows, -1)
	# widthi = np.diff(np.pad(breaksi.ravel(), (1, 0), mode = 'constant', constant_values = -1)).reshape(num_rows, -1)
	max_integer_width = max_integer_width if max_integer_width is not None else widthi.max() - 1
	widthi -= 2

	a0 = np.empty((max_integer_width - 1 + len(a), ), dtype = np.uint8)
	a0[:max_integer_width - 1].fill(0)
	np.subtract(a, np.uint8(ord(b'0')), out = a0[max_integer_width - 1:])
	m = as_strided(a0, shape = [max_integer_width, len(a)], strides = a0.strides * 2)[::-1]
	m = m * np.power(10, np.arange(max_integer_width), dtype = np.int8 if max_integer_width <= 2 else np.int16 if max_integer_width <= 4 else np.int32)[:, None]
	for i in range(1, max_integer_width):
		np.add(m[i], m[i - 1], out = m[i])

	if len(integer_cols) > 0:
		resi = m[widthi[:, integers], breaksi[:, integers]].reshape(num_rows, -1)

	if len(float_cols) > 0:
		BT = breaksi[:, floats].flatten()

		points = a == np.uint8(ord(decimal_point))
		BD = np.flatnonzero(points).astype(np.int32)
		BD -= 1
		BD_BT = np.vstack([BD, BT]).T.flatten()
		
		BF__ = breaksi[:, (floats if not uniform else np.arange(len(float_cols))) - 1].flatten()
		BF__ += 1
		WT = BT - BD - 1
		WD = BD - BF__
		WD_WT = np.vstack([WD, WT]).T.flatten()
		WD_WT -= 1
		WD_WT[WD_WT < 0] = 0

		resf = m[WD_WT, BD_BT].reshape(num_rows, -1)
		resf = np.ascontiguousarray(resf.astype(np.float32).reshape(-1, 2).T)
		np.multiply(resf[1], np.power(10.0, -WT, dtype = np.float32), out = resf[1])
		resf = np.add(resf[0], resf[1], out = resf[1]).reshape(num_rows, -1)

	if not uniform:
		integer_cols, float_cols = [{n : j for j, (i, n) in enumerate(cols)} for cols in [integer_cols, float_cols]]
		return np.rec.fromarrays([resi[:, integer_cols[n]] if n in integer_cols else resf[:, float_cols[n]] for n in head.dtype.names], names=head.dtype.names)
	else:
		return resi if len(integer_cols) > 0 else resf
		
if __name__ == '__main__':
	import gzip, io, time, csv

	if False:
		save_test_case = lambda file_path, x, decimal_width = 4: np.savetxt(file_path, x, delimiter='\t', newline='\n', encoding='utf-8', fmt='\t'.join(dict(i = '%d', f = f'%.{decimal_width}f').get(t[1], '%s') for n, t in x.dtype.descr))
		bounds_int_large = (0, 10000)
		bounds_int_small = (0, 100)
		num_rows = 1000000
		num_cols = 20
		integers = np.random.randint(*bounds_int_large, size = (num_rows, num_cols)).astype(np.int32)
		floats = (np.random.randint(*bounds_int_small, size = (num_rows, num_cols)) + np.random.rand(num_rows, num_cols)).astype(np.float32)

		integers_then_floats = np.rec.fromarrays(list(integers.T) + list(floats.T))
		integers_and_floats = np.rec.fromarrays(c for cc in zip(integers.T, floats.T) for c in cc)

		save_test_case('integers_1000k.txt.gz', integers)
		save_test_case('integers_100k.txt.gz', integers[:100000])
		save_test_case('integers_100.txt', integers[:100])

		save_test_case('floats_1000k.txt.gz', floats)
		save_test_case('floats_100k.txt.gz', floats[:100000])
		save_test_case('floats_100.txt', floats[:100])

		save_test_case('integers_then_floats_1000k.txt.gz', integers_then_floats)
		save_test_case('integers_then_floats_100k.txt.gz', integers_then_floats[:100000])
		save_test_case('integers_then_floats_100.txt', integers_then_floats[:100])

		save_test_case('integers_and_floats_1000k.txt.gz', integers_and_floats)
		save_test_case('integers_and_floats_100k.txt.gz', integers_and_floats[:100000])
		save_test_case('integers_and_floats_100.txt', integers_and_floats[:100])

	def test_case(file_path, delimiter = '\t', encoding = 'utf-8', force_upcast = True, memoryview = None):
		print()
		print('Test case:', file_path, 'force_upcast', force_upcast)
		f = (gzip.open if file_path.endswith('.gz') else open)(file_path , 'rb') if memoryview is None else gzip.open(io.BytesIO(memoryview), 'rb')
		b = f.read()
		s = b.decode('ascii')
		dtype = [(n, t.replace('8', '4')) for n, t in np.genfromtxt(io.BytesIO(b), max_rows = 1, delimiter = delimiter, names = None, dtype = None, encoding = encoding).dtype.descr]
		uniform = len(dtype) == 1
		dtype = dtype[0][1] if uniform else dtype

		tic = time.time()
		x = np.loadtxt(io.StringIO(s), delimiter = delimiter, dtype = np.float32 if not uniform and force_upcast else dtype)
		print('loadtxt', time.time() - tic)

		tic = time.time()
		y = np.fromstring(s, count = len(x) * len(x[0]), dtype = np.float32 if not uniform else dtype, sep = delimiter).reshape(len(x), len(x[0]))
		print('fromstring', time.time() - tic)

		tic = time.time()
		list(csv.reader(io.StringIO(s), delimiter = delimiter, quoting = csv.QUOTE_NONNUMERIC))
		print('csv.reader', time.time() - tic)

		tic = time.time()
		loads(b, max_integer_width = 4)
		print('fasttsv', time.time() - tic)
		print()

	#test_case('integers_100k.txt.gz')
	#test_case('floats_100k.txt.gz')

	#test_case('integers_then_floats_100k.txt', force_upcast = True)
	#test_case('integers_then_floats_100k.txt', force_upcast = False)
	#test_case('integers_and_floats_100k.txt', force_upcast = True)

	# python3 -m cProfile -s cumtime fasttsv.py

	#tic = time.time()
	#print(loads(b'# a\tb\tc\td\n1\t22.60\t3\t5.0\n3\t44.8\t8\t9.09\n'))

	#print(tsvparse(b'1\t22\n3\t44\n', integers = True))
	#np.savetxt('test.txt', np.random.randint(0, 10000, size = (100000, 20)),fmt = '%d', delimiter = '\t')
	#np.savetxt('test2.txt', np.random.rand(100000, 20) * 10,fmt = '%.4f', delimiter = '\t')
	#np.savetxt('test3.txt', np.random.randint(0, 10000, size = (1000000, 20)),fmt = '%d', delimiter = '\t')
	
	# baseline
	#res = np.loadtxt(io.StringIO(b.decode('ascii')), dtype = np.float32 if floats else int, delimiter = '\t')

	b = open('floats_100k.txt', 'rb').read()
	tic = time.time()
	import timeit
	print(timeit.timeit('loads(b, max_integer_width = 4)', number = 10, globals = globals()) / 10)
	#print(loads(b, max_integer_width = 4))
	#print(time.time() - tic)

	#list(csv.reader(open('test3.txt'), delimiter = '\t'))
	#M = loads(open('test.txt', 'rb').read()); import IPython; IPython.embed()

