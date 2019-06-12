import numpy as np
from numpy.lib.stride_tricks import as_strided

def loads(b, max_integer_width = None, encoding = 'utf-8', delimiter = '\t', newline = '\n', comments = '#', decimal_point = '.', force_upcast = False):
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
	num_rows = newlines.sum()

	def downcast(integer_array, other = None):
		max = integer_array.max()
		if other is not None and max < np.iinfo(other.dtype).max:
			return integer_array.astype(other.dtype)
		for dt in [np.int8, np.int16, np.int32]:
			if max < np.iinfo(dt).max:
				return integer_array.astype(dt, copy = False)
		return integer_array

	breaksi = downcast(np.flatnonzero(np.bitwise_or(tabs, newlines, out = tabs)).reshape(num_rows, -1)); breaksi -= 1 # last character of integers
	widthi = np.empty(breaksi.shape, dtype = np.int8)
	np.subtract(breaksi.ravel()[1:], breaksi.ravel()[:-1], out = widthi.ravel()[1:])
	widthi[0, 0] = breaksi[0, 0] + 2
	widthi -= 2 # width of integers - 1

	max_integer_width = max_integer_width if max_integer_width is not None else widthi.max() + 1

	a0 = np.empty((max_integer_width - 1 + len(a), ), dtype = np.int8)
	a0[:max_integer_width - 1].fill(0)
	np.subtract(a, np.uint8(ord(b'0')), out = a0[max_integer_width - 1:])
	m = as_strided(a0, shape = [max_integer_width, len(a)], strides = a0.strides * 2)[::-1]
	m = m * np.power(10, np.arange(max_integer_width), dtype = np.int8 if max_integer_width <= 2 else np.int16 if max_integer_width <= 4 else np.int32)[:, None]
	for i in range(1, max_integer_width):
		np.add(m[i], m[i - 1], out = m[i])

	if len(integer_cols) > 0:
		resi = m[widthi[:, integers], breaksi[:, integers]].reshape(num_rows, -1)

	if len(float_cols) > 0:
		points = a == np.uint8(ord(decimal_point))
		BT = breaksi[:, floats] # last character of remainder
		BD = downcast(np.flatnonzero(points)) # dots indices
		BD = np.subtract(BD, 1, out = BD).reshape(BT.shape) # last character of integral part
		WT = np.subtract(BT, BD, dtype = np.int8); WT -= 1  # width of remainder

		resf = np.power(10.0, -WT, dtype = np.float32)
		WT -= 1; resf *= m[WT, BT]
		WD = np.subtract(widthi[:, floats], WT, out = WT)
		WD -= 2; resf += m[WD, BD]
		resf = resf.reshape(num_rows, -1)

	if uniform:
		return resf if len(float_cols) > 0 else resi
	else:
		integer_cols, float_cols = [{n : j for j, (i, n) in enumerate(cols)} for cols in [integer_cols, float_cols]]

		if force_upcast:
			dtype = resf.dtype if len(float_cols) > 0 else resi.dtype
			return np.column_stack([(resi[:, integer_cols[n]] if n in integer_cols else resf[:, float_cols[n]]).astype(dtype) for n in head.dtype.names])
		else:
			return np.rec.fromarrays([resi[:, integer_cols[n]] if n in integer_cols else resf[:, float_cols[n]] for n in head.dtype.names], names=head.dtype.names)
		
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
		y = loads(b, max_integer_width = 4, force_upcast = force_upcast)
		print('fasttsv', time.time() - tic)
		if force_upcast:
			print('max-abs-diff', np.abs(x - y).max())
		print()

	test_case('integers_100k.txt.gz')
	test_case('floats_100k.txt.gz')
	#test_case('integers_and_floats_100k.txt', force_upcast = True)
	#test_case('integers_then_floats_100k.txt', force_upcast = True)
	#test_case('integers_then_floats_100k.txt', force_upcast = False)

	# python3 -m cProfile -s cumtime fasttsv.py

	#b = open('integers_and_floats_100.txt', 'rb').read()
	#import timeit; #print(timeit.timeit('loads(b, max_integer_width = 4)', number = 10, globals = globals()) / 10)
	#print(loads(b'123.567\t2.45\t4\n', max_integer_width = 4))
	#print(loads(b, max_integer_width = 4, force_upcast = True))
