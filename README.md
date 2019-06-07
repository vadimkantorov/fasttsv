# fasttsv
TSV parser for Python in pure NumPy vector code.

This is not production-ready code, it is mostly a primer on a branchless parsing technique using vectorized code (inspired by [simdjson](https://github.com/lemire/simdjson) and [csvmonkey](https://github.com/dw/csvmonkey)).

# Approach
1. Read the whole file into a byte array in memory
2. Find positions of tabs and decimal points
3. Compute digit count for every field
3. For the integer case, given the maximum number of digits in the file, precompute the parsed integers finishing on a given positions for all possible digit counts
4. For every field, use the computed digit count to index into the precomputed parsed integers array
5. Assemble values for the real-valued columns: the integral and remainder parts are neighboring parsed integers

# Features, scope and limitations
1. Supports integer, real (only decimal point notation) and utf-8 string columns (quotes not supported)
2. Uses only NumPy methods, and can be extended to GPU using Google Jax or PyTorch. It can also run on Pyodide.
