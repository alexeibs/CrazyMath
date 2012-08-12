#ifndef CrazyMath_h
#define CrazyMath_h

#include <cmath>

namespace CrazyMath {

//---------------------------------------------------
// base functions

class Const {
public:
	Const(double x)
		: m_const(x)
	{
	}
	Const(int x)
		: m_const(x)
	{
	}
	double operator()(double)
	{
		return m_const;
	}
	operator double()
	{
		return m_const;
	}
private:
	double m_const;
};

class Simple {
public:
	double operator()(double x)
	{
		return x;
	}
};

template <class F1, class F2>
class Add {
public:
	Add(const F1& f1, const F2& f2)
		: m_f1(f1), m_f2(f2)
	{
	}
	double operator()(double x)
	{
		return m_f1(x) + m_f2(x);
	}
	F1 m_f1;
	F2 m_f2;
};

template <class F1, class F2>
class Subtract {
public:
	Subtract(const F1& f1, const F2& f2)
		: m_f1(f1), m_f2(f2)
	{
	}
	double operator()(double x)
	{
		return m_f1(x) - m_f2(x);
	}
	F1 m_f1;
	F2 m_f2;
};

template <class F1, class F2>
class Multiply {
public:
	Multiply(const F1& f1, const F2& f2)
		: m_f1(f1), m_f2(f2)
	{
	}
	double operator()(double x)
	{
		return m_f1(x) * m_f2(x);
	}
	F1 m_f1;
	F2 m_f2;
};

template <class F1, class F2>
class Divide {
public:
	Divide(const F1& f1, const F2& f2)
		: m_f1(f1), m_f2(f2)
	{
	}
	double operator()(double x)
	{
		return m_f1(x) / m_f2(x);
	}
	F1 m_f1;
	F2 m_f2;
};

template <class F>
class Power {
public:
	Power(const F& f, double n)
		: m_f(f), m_n(n)
	{
	}
	double operator()(double x)
	{
		return pow(m_f(x), m_n);
	}
	F m_f;
	double m_n;
};

template <class F>
class Exponent {
public:
	Exponent(double base, const F& f)
		: m_base(base), m_f(f)
	{
	}
	double operator()(double x)
	{
		return pow(m_base, m_f(x));
	}
	double m_base;
	F m_f;
};

template <class F>
class Logarithm {
public:
	Logarithm(double base, const F& f)
		: m_base(base), m_factor(1 / log(base)), m_f(f)
	{
	}
	double operator()(double x)
	{
		return log(m_f(x)) * m_factor;
	}
	double m_base, m_factor;
	F m_f;
};

template <class F>
class Sine {
public:
	Sine(const F& f)
		: m_f(f)
	{
	}
	double operator()(double x)
	{
		return sin(m_f(x));
	}
	F m_f;
};

template <class F>
class Cosine {
public:
	Cosine(const F& f)
		: m_f(f)
	{
	}
	double operator()(double x)
	{
		return cos(m_f(x));
	}
	F m_f;
};

template <class F>
class Tangent {
public:
	Tangent(const F& f)
		: m_f(f)
	{
	}
	double operator()(double x)
	{
		return tan(m_f(x));
	}
	F m_f;
};

template <class F>
class Cotangent {
public:
	Cotangent(const F& f)
		: m_f(f)
	{
	}
	double operator()(double x)
	{
		return 1 / tan(m_f(x));
	}
	F m_f;
};

template <class F>
class Arcsine {
public:
	Arcsine(const F& f)
		: m_f(f)
	{
	}
	double operator()(double x)
	{
		return asin(m_f(x));
	}
	F m_f;
};

template <class F>
class Arccosine {
public:
	Arccosine(const F& f)
		: m_f(f)
	{
	}
	double operator()(double x)
	{
		return acos(m_f(x));
	}
	F m_f;
};

template <class F>
class Arctangent {
public:
	Arctangent(const F& f)
		: m_f(f)
	{
	}
	double operator()(double x)
	{
		return atan(m_f(x));
	}
	F m_f;
};

template <class F>
class Arccotangent {
public:
	Arccotangent(const F& f)
		: m_f(f)
	{
	}
	double operator()(double x)
	{
		return atan(1 / m_f(x));
	}
	F m_f;
};

//---------------------------------------------------
// derivatives

template <class F>
class Derivative {
public:
	Derivative(const F& f, double dx = 1e-3)
		: m_f(f), m_dx(dx)
	{
	}
	double operator()(double x)
	{
		return (m_f(x + m_dx) - m_f(x)) / 1e-3;
	}
	F m_f;
	double m_dx;
};

template<>
class Derivative<Const> {
public:
	Derivative<Const> (Const) {}
	double operator()(double)
	{
		return 0;
	}
};

template<>
class Derivative<Simple> {
public:
	Derivative<Simple> (Simple) {}
	double operator()(double)
	{
		return 1;
	}
};

template <class F1, class F2>
class Derivative< Add<F1, F2> > {
public:
	Derivative< Add<F1, F2> > (const Add<F1, F2>& f)
		: m_df1(f.m_f1), m_df2(f.m_f2)
	{
	}
	double operator()(double x)
	{
		return m_df1(x) + m_df2(x);
	}
	Derivative<F1> m_df1;
	Derivative<F2> m_df2;
};

template <class F1, class F2>
class Derivative< Subtract<F1, F2> > {
public:
	Derivative< Subtract<F1, F2> > (const Subtract<F1, F2>& f)
		: m_df1(f.m_f1), m_df2(f.m_f2)
	{
	}
	double operator()(double x)
	{
		return m_df1(x) - m_df2(x);
	}
	Derivative<F1> m_df1;
	Derivative<F2> m_df2;
};

template <class F1, class F2>
class Derivative< Multiply<F1, F2> > {
public:
	Derivative< Multiply<F1, F2> > (const Multiply<F1, F2>& f)
		: m_f1(f.m_f1), m_f2(f.m_f2), m_df1(f.m_f1), m_df2(f.m_f2)
	{
	}
	double operator()(double x)
	{
		return m_df1(x) * m_f2(x) + m_f1(x) * m_df2(x);
	}
	F1 m_f1;
	F2 m_f2;
	Derivative<F1> m_df1;
	Derivative<F2> m_df2;
};

template <class F1, class F2>
class Derivative< Divide<F1, F2> > {
public:
	Derivative< Divide<F1, F2> > (const Divide<F1, F2>& f)
		: m_f1(f.m_f1), m_f2(f.m_f2), m_df1(f.m_f1), m_df2(f.m_f2)
	{
	}
	double operator()(double x)
	{
		double f2 = m_f2(x);
		return (m_df1(x) * f2 - m_f1(x) * m_df2(x)) / (f2 * f2);
	}
	F1 m_f1;
	F2 m_f2;
	Derivative<F1> m_df1;
	Derivative<F2> m_df2;
};

template <class F>
class Derivative< Power<F> > {
public:
	Derivative< Power<F> > (const Power<F>& f)
		: m_f(f.m_f), m_n(f.m_n), m_df(f.m_f)
	{
	}
	double operator()(double x)
	{
		return m_n * pow(m_f(x), m_n - 1) * m_df(x);
	}
	F m_f;
	double m_n;
	Derivative<F> m_df;
};

template <class F>
class Derivative< Exponent<F> > {
public:
	Derivative< Exponent<F> >(const Exponent<F>& f)
		: m_base(f.m_base), m_factor(log(f.m_base))
		, m_f(f.m_f), m_df(f.m_f)
	{
	}
	double operator()(double x)
	{
		return pow(m_base, m_f(x)) * m_factor * m_df(x);
	}
	double m_base, m_factor;
	F m_f;
	Derivative<F> m_df;
};

template <class F>
class Derivative< Logarithm<F> > {
public:
	Derivative< Logarithm<F> >(const Logarithm<F>& f)
		: m_base(f.m_base), m_factor(1 / log(m_base)), m_f(f.m_f), m_df(f.m_f)
	{
	}
	double operator()(double x)
	{
		return m_factor * m_df(x) / x;
	}
	double m_base, m_factor;
	F m_f;
	Derivative<F> m_df;
};

template <class F>
class Derivative< Sine<F> > {
public:
	Derivative< Sine<F> >(const Sine<F>& f)
		: m_f(f.m_f), m_df(f.m_f)
	{
	}
	double operator()(double x)
	{
		return cos(m_f(x)) * m_df(x);
	}
	F m_f;
	Derivative<F> m_df;
};

template <class F>
class Derivative< Cosine<F> > {
public:
	Derivative< Cosine<F> >(const Cosine<F>& f)
		: m_f(f.m_f), m_df(f.m_f)
	{
	}
	double operator()(double x)
	{
		return -sin(m_f(x)) * m_df(x);
	}
	F m_f;
	Derivative<F> m_df;
};

template <class F>
class Derivative< Tangent<F> > {
public:
	Derivative< Tangent<F> >(const Tangent<F>& f)
		: m_f(f.m_f), m_df(f.m_f)
	{
	}
	double operator()(double x)
	{
		double cosfx = cos(m_f(x));
		return m_df(x) / (cosfx * cosfx);
	}
	F m_f;
	Derivative<F> m_df;
};

template <class F>
class Derivative< Cotangent<F> > {
public:
	Derivative< Cotangent<F> >(const Cotangent<F>& f)
		: m_f(f.m_f), m_df(f.m_f)
	{
	}
	double operator()(double x)
	{
		double sinfx = sin(m_f(x));
		return -m_df(x) / (sinfx * sinfx);
	}
	F m_f;
	Derivative<F> m_df;
};

template <class F>
class Derivative< Arcsine<F> > {
public:
	Derivative< Arcsine<F> >(const Arcsine<F>& f)
		: m_f(f.m_f), m_df(f.m_f)
	{
	}
	double operator()(double x)
	{
		double fx = m_f(x);
		return m_df(x) / sqrt(1 - fx * fx);
	}
	F m_f;
	Derivative<F> m_df;
};

template <class F>
class Derivative< Arccosine<F> > {
public:
	Derivative< Arccosine<F> >(const Arccosine<F>& f)
		: m_f(f.m_f), m_df(f.m_f)
	{
	}
	double operator()(double x)
	{
		double fx = m_f(x);
		return -m_df(x) / sqrt(1 - fx * fx);
	}
	F m_f;
	Derivative<F> m_df;
};

template <class F>
class Derivative< Arctangent<F> > {
public:
	Derivative< Arctangent<F> >(const Arctangent<F>& f)
		: m_f(f.m_f), m_df(f.m_f)
	{
	}
	double operator()(double x)
	{
		double fx = m_f(x);
		return m_df(x) / (1 + fx * fx);
	}
	F m_f;
	Derivative<F> m_df;
};

template <class F>
class Derivative< Arccotangent<F> > {
public:
	Derivative< Arccotangent<F> >(const Arccotangent<F>& f)
		: m_f(f.m_f), m_df(f.m_f)
	{
	}
	double operator()(double x)
	{
		double fx = m_f(x);
		return -m_df(x) / (1 + fx * fx);
	}
	F m_f;
	Derivative<F> m_df;
};

//---------------------------------------------------
// helpers

template <class F1, class F2>
Add<F1, F2> operator+(const F1& f1, const F2& f2)
{
	return Add<F1, F2>(f1, f2);
}

template <class F>
Add<F, Const> operator+(double value, const F& f)
{
	return Add<F, Const>(f, Const(value));
}

template <class F>
Add<F, Const> operator+(const F& f, int value)
{
	return Add<F, Const>(f, Const(value));
}

template <class F>
Add<F, Const> operator+(int value, const F& f)
{
	return Add<F, Const>(f, Const(value));
}

template <class F>
Add<F, Const> operator+(const F& f, double value)
{
	return Add<F, Const>(f, Const(value));
}

template <class F1, class F2>
Subtract<F1, F2> operator-(const F1& f1, const F2& f2)
{
	return Subtract<F1, F2>(f1, f2);
}

template <class F>
Subtract<F, Const> operator-(const F& f, double value)
{
	return Subtract<F, Const>(f, Const(value));
}

template <class F>
Subtract<Const, F> operator-(double value, const F& f)
{
	return Subtract<Const, F>(Const(value), f);
}

template <class F>
Subtract<F, Const> operator-(const F& f, int value)
{
	return Subtract<F, Const>(f, Const(value));
}

template <class F>
Subtract<Const, F> operator-(int value, const F& f)
{
	return Subtract<Const, F>(Const(value), f);
}

template <class F1, class F2>
Multiply<F1, F2> operator*(const F1& f1, const F2& f2)
{
	return Multiply<F1, F2>(f1, f2);
}

template <class F>
Multiply<F, Const> operator*(const F& f, double value)
{
	return Multiply<F, Const>(f, Const(value));
}

template <class F>
Multiply<F, Const> operator*(double value, const F& f)
{
	return Multiply<F, Const>(f, Const(value));
}

template <class F>
Multiply<F, Const> operator*(const F& f, int value)
{
	return Multiply<F, Const>(f, Const(value));
}

template <class F>
Multiply<F, Const> operator*(int value, const F& f)
{
	return Multiply<F, Const>(f, Const(value));
}

template <class F1, class F2>
Divide<F1, F2> operator/(const F1& f1, const F2& f2)
{
	return Divide<F1, F2>(f1, f2);
}

template <class F>
Divide<F, Const> operator/(const F& f, double value)
{
	return Divide<F, Const>(f1, Const(value));
}

template <class F>
Divide<Const, F> operator/(double value, const F& f)
{
	return Divide<Const, F>(Const(value), f);
}

template <class F>
Divide<F, Const> operator/(const F& f, int value)
{
	return Divide<F, Const>(f, Const(value));
}

template <class F>
Divide<Const, F> operator/(int value, const F& f)
{
	return Divide<Const, F>(Const(value), f);
}

template <class F>
Multiply<F, F> Sqr(const F& f)
{
	return Multiply<F, F>(f, f);
}

inline double Sqr(double x)
{
	return x * x;
}

template <class F, class Numeric>
Power<F> Pow(const F& f, Numeric n)
{
	return Power<F>(f, n);
}

inline double Pow(double x, double y)
{
	return pow(x, y);
}

template <class F>
Power<F> Sqrt(const F& f)
{
	return Power<F>(f, 0.5);
}

inline double Sqrt(double x)
{
	return sqrt(x);
}

template <class F>
Exponent<F> Exp(double base, const F& f)
{
	return Exponent<F>(base, f);
}

template <class F>
Exponent<F> Exp(int base, const F& f)
{
	return Exponent<F>(base, f);
}

inline double Exp(double base, double x)
{
	return pow(base, x);
}

template <class F>
Logarithm<F> Log(double base, const F& f)
{
	return Logarithm<F>(base, f);
}

template <class F>
Logarithm<F> Log(int base, const F& f)
{
	return Logarithm<F>(base, f);
}

inline double Log(double base, double x)
{
	return log(x) / log(base);
}

template <class F>
Sine<F> Sin(const F& f)
{
	return Sine<F>(f);
}

inline double Sin(double x)
{
	return sin(x);
}

template <class F>
Cosine<F> Cos(const F& f)
{
	return Cosine<F>(f);
}

inline double Cos(double x)
{
	return cos(x);
}

template <class F>
Tangent<F> Tg(const F& f)
{
	return Tangent<F>(f);
}

inline double Tg(double x)
{
	return tan(x);
}

template <class F>
Cotangent<F> Ctg(const F& f)
{
	return Cotangent<F>(f);
}

inline double Ctg(double x)
{
	return 1 / tan(x);
}

template <class F>
Arcsine<F> Asin(const F& f)
{
	return Arcsine<F>(f);
}

inline double Asin(double x)
{
	return asin(x);
}

template <class F>
Arccosine<F> Acos(const F& f)
{
	return Arccosine<F>(f);
}

inline double Acos(double x)
{
	return acos(x);
}

template <class F>
Arctangent<F> Atg(const F& f)
{
	return Arctangent<F>(f);
}

inline double Atg(double x)
{
	return atan(x);
}

template <class F>
Arccotangent<F> Actg(const F& f)
{
	return Arccotangent<F>(f);
}

inline double Actg(double x)
{
	return atan(1 / x);
}

template <class F>
Derivative<F> derivative(F f)
{
	return Derivative<F>(f);
}

extern Simple X;

}

#endif // CrazyMath_h