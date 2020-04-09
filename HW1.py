import sympy as sp
import numpy as np
from numpy import linalg as la
import mpmath
import math
from matplotlib import pyplot as plt
import time

def dn_sin(n):
	'''
	Compute the n^th derivative of sin(x) at x=0

	input:
		n - int: the order of the derivative to compute
	output:
		float nth derivative of sin(0)
	'''
	if n%4 == 0:
		return sp.sin(0)
	elif n%4 == 1:
		return sp.cos(0)
	elif n%4 == 2:
		return sp.sin(0)*(-1)
	else:
		return sp.cos(0)*(-1)


def fact(n):
	f = 1
	for i in range(1,n+1):
		f = f * i
	return f


def taylor_sin(x, n):
	'''
	Evaluate the Taylor series of sin(x) about x=0 neglecting terms of order x^n
	
	input:
		x - float: argument of sin
		n - int: number of terms of the taylor series to use in approximation
	output:
		float value computed using the taylor series truncated at the nth term
	'''
	sum_t = 0
	for i in range(0,n+1):
		sum_t = sum_t + (dn_sin(i)*(x**i))/fact(i)
	return sum_t


def measure_diff(ary1, ary2):
	'''
	Compute a scalar measure of difference between 2 arrays

	input:
		ary1 - numpy array of float values
		ary2 - numpy array of float values
	output:
		a float scalar quantifying difference between the arrays
	'''
	a_diff = np.subtract(ary1,ary2)
	n_diff = la.norm(a_diff)
	return n_diff


def escape(cx, cy, dist,itrs, x0=0, y0=0):
	'''
	Compute the number of iterations of the logistic map, 
	f(x+j*y)=(x+j*y)**2 + cx +j*cy with initial values x0 and y0 
	with default values of 0, to escape from a cirle centered at the origin.

	inputs:
		cx - float: the real component of the parameter value
		cy - float: the imag component of the parameter value
		dist: radius of the circle
		itrs: int max number of iterations to compute
		x0: initial value of x; default value 0
		y0: initial value of y; default value 0
	returns:
		an int scalar iteration count
	'''
	z = complex(x0,y0)
	for i in range(itrs):
		zx = z.real
		zy = z.imag
		z = z*z + complex(cx,cy)
		zx = z.real
		zy = z.imag
		if la.norm([zx,zy]) > dist:
			break
	return i


def mandelbrot(cx,cy,dist,itrs):
	'''
	Compute escape iteration counts for an array of parameter values

	input:
		cx - array: 1d array of real part of parameter
		cy - array: 1d array of imaginary part of parameter
		dist - float: radius of circle for escape
		itrs - int: maximum number of iterations to compute
	output:
		a 2d array of iteration count for each parameter value (indexed pair of values cx, cy)
	'''
	lex = len(cx)
	ley = len(cy)
	ary = np.zeros((lex,ley))
	for i in range(ley):
		for j in range(lex):
			ary[i,j] = escape(cx[j],cy[i],2.5,256)
	return ary

if __name__ == '__main__':


	#Problem 3
	#compute taylor series
	#plot taylor series
	#`pass` prevents an error message if you run this file before inserting code
	x = np.linspace(-5,5)
	y = np.zeros(50)
	for i in range(0,50):
		y[i] = sp.sin(x[i])
	plt.plot(x,y,linewidth = 4)
	for n in range(2,16,2):
		for j in range(0,50):
			y[j] = taylor_sin(x[j], i)
		plt.plot(x,y)
	plt.legend(['sin(x)','n=2','n=4','n=6','n=8','n=10','n=12','n=14'])
		

	#Problem 4
	x = np.linspace(0,math.pi/4)
	y_sin = np.zeros(50)
	y_tay = np.zeros(50)
	for i in range(0,50):
		y_sin[i] = sp.sin(x[i])
	for n in range(2,16,2):
		for j in range(0,50):
			y_tay[j] = taylor_sin(x[j], i)
		d = measure_diff(y_sin,y_tay)
		if d < 1e-2:
			print('The truncation order needed is n = '+ str(i))
			break
		

	#Problem 5
	cx = np.linspace(-2.5,2.5,512)
	cy = np.linspace(-2.5,2.5,512)
	lex = len(cx)
	ley = len(cy)
	start = time.time()
	bo_ary = mandelbrot(cx,cy,2.5,256)
	end = time.time()
	for i in range(lex):
		for j in range(ley):
			if bo_ary[i,j] < 255:
				bo_ary[i,j] = 1
			else:
				bo_ary[i,j] = 0
	exe = end - start
	print('The execution time is '+ str(exe) + ' seconds')
	plt.pcolormesh(cx,cy,bo_ary)
