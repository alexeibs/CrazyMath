#ifndef CrazyMath_h
#define CrazyMath_h

#include <cmath>
#include <functional>
#include <type_traits>

namespace CrazyMath {

//---------------------------------------------------
// base functions

class Const {
public:
	typedef Const Type;
	Const(double x)
		: m_const(x)
	{
	}
	Const(int x)
		: m_const(x)
	{
	}
	double operator()(double) const
	{
		return m_const;
	}
	operator double()
	{
		return m_const;
	}
	double m_const;
};

class Simple {
public:
	typedef Simple Type;
	double operator()(double x) const
	{
		return x;
	}
};

template <class F1, class F2>
class Add {
public:
	typedef Add<F1, F2> Type;
	Add(const F1& f1, const F2& f2)
		: m_f1(f1), m_f2(f2)
	{
	}
	double operator()(double x) const
	{
		return m_f1(x) + m_f2(x);
	}
	F1 m_f1;
	F2 m_f2;
};

template <class F1, class F2>
class Subtract {
public:
	typedef Subtract<F1, F2> Type;
	Subtract(const F1& f1, const F2& f2)
		: m_f1(f1), m_f2(f2)
	{
	}
	double operator()(double x) const
	{
		return m_f1(x) - m_f2(x);
	}
	F1 m_f1;
	F2 m_f2;
};

template <class F1, class F2>
class Multiply {
public:
	typedef Multiply<F1, F2> Type;
	Multiply(const F1& f1, const F2& f2)
		: m_f1(f1), m_f2(f2)
	{
	}
	double operator()(double x) const
	{
		return m_f1(x) * m_f2(x);
	}
	F1 m_f1;
	F2 m_f2;
};

template <class F1, class F2>
class Divide {
public:
	typedef Divide<F1, F2> Type;
	Divide(const F1& f1, const F2& f2)
		: m_f1(f1), m_f2(f2)
	{
	}
	double operator()(double x) const
	{
		return m_f1(x) / m_f2(x);
	}
	F1 m_f1;
	F2 m_f2;
};

template <class F>
class Power {
public:
	typedef Power<F> Type;
	Power(const F& f, double n)
		: m_f(f), m_n(n)
	{
	}
	double operator()(double x) const
	{
		return pow(m_f(x), m_n);
	}
	F m_f;
	double m_n;
};

template <class F>
class Exponent {
public:
	typedef Exponent<F> Type;
	Exponent(double base, const F& f)
		: m_base(base), m_f(f)
	{
	}
	double operator()(double x) const
	{
		return pow(m_base, m_f(x));
	}
	double m_base;
	F m_f;
};

template <class F>
class Logarithm {
public:
	typedef Logarithm<F> Type;
	Logarithm(double base, const F& f)
		: m_base(base), m_factor(1 / log(base)), m_f(f)
	{
	}
	double operator()(double x) const
	{
		return log(m_f(x)) * m_factor;
	}
	double m_base, m_factor;
	F m_f;
};

template <class F>
class Sine {
public:
	typedef Sine<F> Type;
	Sine(const F& f)
		: m_f(f)
	{
	}
	double operator()(double x) const
	{
		return sin(m_f(x));
	}
	F m_f;
};

template <class F>
class Cosine {
public:
	typedef Cosine<F> Type;
	Cosine(const F& f)
		: m_f(f)
	{
	}
	double operator()(double x) const
	{
		return cos(m_f(x));
	}
	F m_f;
};

template <class F>
class Tangent {
public:
	typedef Tangent<F> Type;
	Tangent(const F& f)
		: m_f(f)
	{
	}
	double operator()(double x) const
	{
		return tan(m_f(x));
	}
	F m_f;
};

template <class F>
class Cotangent {
public:
	typedef Cotangent<F> Type;
	Cotangent(const F& f)
		: m_f(f)
	{
	}
	double operator()(double x) const
	{
		return 1 / tan(m_f(x));
	}
	F m_f;
};

template <class F>
class Arcsine {
public:
	typedef Arcsine<F> Type;
	Arcsine(const F& f)
		: m_f(f)
	{
	}
	double operator()(double x) const
	{
		return asin(m_f(x));
	}
	F m_f;
};

template <class F>
class Arccosine {
public:
	typedef Arccosine<F> Type;
	Arccosine(const F& f)
		: m_f(f)
	{
	}
	double operator()(double x) const
	{
		return acos(m_f(x));
	}
	F m_f;
};

template <class F>
class Arctangent {
public:
	typedef Arctangent<F> Type;
	Arctangent(const F& f)
		: m_f(f)
	{
	}
	double operator()(double x) const
	{
		return atan(m_f(x));
	}
	F m_f;
};

template <class F>
class Arccotangent {
public:
	typedef Arccotangent<F> Type;
	Arccotangent(const F& f)
		: m_f(f)
	{
	}
	double operator()(double x) const
	{
		return atan(1 / m_f(x));
	}
	F m_f;
};

//---------------------------------------------------
// helpers

template <class F1, class F2>
Add<F1, F2> operator+(const F1& f1, const F2& f2)
{
	return Add<F1, F2>(f1, f2);
}

template <class F>
typename std::enable_if<!std::is_arithmetic<F>::value, Add<F, Const>>::type operator+(double value, const F& f)
{
	return Add<F, Const>(f, Const(value));
}

template <class F>
typename std::enable_if<!std::is_arithmetic<F>::value, Add<F, Const>>::type operator+(const F& f, int value)
{
	return Add<F, Const>(f, Const(value));
}

template <class F>
typename std::enable_if<!std::is_arithmetic<F>::value, Add<F, Const>>::type operator+(int value, const F& f)
{
	return Add<F, Const>(f, Const(value));
}

template <class F>
typename std::enable_if<!std::is_arithmetic<F>::value, Add<F, Const>>::type operator+(const F& f, double value)
{
	return Add<F, Const>(f, Const(value));
}

template <class F1, class F2>
Subtract<F1, F2> operator-(const F1& f1, const F2& f2)
{
	return Subtract<F1, F2>(f1, f2);
}

template <class F>
typename std::enable_if<!std::is_arithmetic<F>::value, Subtract<F, Const>>::type operator-(const F& f, double value)
{
	return Subtract<F, Const>(f, Const(value));
}

template <class F>
typename std::enable_if<!std::is_arithmetic<F>::value, Subtract<Const, F>>::type operator-(double value, const F& f)
{
	return Subtract<Const, F>(Const(value), f);
}

template <class F>
typename std::enable_if<!std::is_arithmetic<F>::value, Subtract<F, Const>>::type operator-(const F& f, int value)
{
	return Subtract<F, Const>(f, Const(value));
}

template <class F>
typename std::enable_if<!std::is_arithmetic<F>::value, Subtract<Const, F>>::type operator-(int value, const F& f)
{
	return Subtract<Const, F>(Const(value), f);
}

template <class F1, class F2>
Multiply<F1, F2> operator*(const F1& f1, const F2& f2)
{
	return Multiply<F1, F2>(f1, f2);
}

template <class F>
typename std::enable_if<!std::is_arithmetic<F>::value, Multiply<F, Const>>::type operator*(const F& f, double value)
{
	return Multiply<F, Const>(f, Const(value));
}

template <class F>
typename std::enable_if<!std::is_arithmetic<F>::value, Multiply<F, Const>>::type operator*(double value, const F& f)
{
	return Multiply<F, Const>(f, Const(value));
}

template <class F>
typename std::enable_if<!std::is_arithmetic<F>::value, Multiply<F, Const>>::type operator*(const F& f, int value)
{
	return Multiply<F, Const>(f, Const(value));
}

template <class F>
typename std::enable_if<!std::is_arithmetic<F>::value, Multiply<F, Const>>::type operator*(int value, const F& f)
{
	return Multiply<F, Const>(f, Const(value));
}

template <class F1, class F2>
Divide<F1, F2> operator/(const F1& f1, const F2& f2)
{
	return Divide<F1, F2>(f1, f2);
}

template <class F>
typename std::enable_if<!std::is_arithmetic<F>::value, Divide<F, Const>>::type operator/(const F& f1, double value)
{
	return Divide<F, Const>(f1, Const(value));
}

template <class F>
typename std::enable_if<!std::is_arithmetic<F>::value, Divide<Const, F>>::type operator/(double value, const F& f)
{
	return Divide<Const, F>(Const(value), f);
}

template <class F>
typename std::enable_if<!std::is_arithmetic<F>::value, Divide<F, Const>>::type operator/(const F& f, int value)
{
	return Divide<F, Const>(f, Const(value));
}

template <class F>
typename std::enable_if<!std::is_arithmetic<F>::value, Divide<Const, F>>::type operator/(int value, const F& f)
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

extern Simple X;

//---------------------------------------------------
// derivatives

template <class F>
class Derivative {
public:
	Derivative(const F& f, double dx = 1e-3)
		: m_f(f), m_dx(dx)
	{
	}
	double operator()(double x) const
	{
		return (m_f(x + m_dx) - m_f(x)) / m_dx;
	}
	F m_f;
	double m_dx;
	typedef std::function<double (double)> Type;
	Type expression() const
	{
		return [this](double x) -> double
		{
			return (m_f(x + m_dx) - m_f(x)) / m_dx;
		};
	}
};

template<>
class Derivative<Const> {
public:
	typedef Const Type;
	Derivative<Const> (Const) {}
	double operator()(double) const
	{
		return 0;
	}
	Type expression() const
	{
		return Const(0);
	}
};

template<>
class Derivative<Simple> {
public:
	typedef Const Type;
	Derivative<Simple> (Simple) {}
	double operator()(double) const
	{
		return 1;
	}
	Type expression() const
	{
		return Const(1);
	}
};

template <class F1, class F2>
class Derivative< Add<F1, F2> > {
public:
	Derivative< Add<F1, F2> > (const Add<F1, F2>& f)
		: m_df1(f.m_f1), m_df2(f.m_f2)
	{
	}
	double operator()(double x) const
	{
		return m_df1(x) + m_df2(x);
	}
	Derivative<F1> m_df1;
	Derivative<F2> m_df2;
	typedef Add<typename Derivative<F1>::Type, typename Derivative<F2>::Type> Type;
	Type expression() const
	{
		return m_df1.expression() + m_df2.expression();
	}
};

template <class F1>
class Derivative< Add<F1, Const> > {
public:
	Derivative< Add<F1, Const> > (const Add<F1, Const>& f)
		: m_df1(f.m_f1)
	{
	}
	double operator()(double x) const
	{
		return m_df1(x);
	}
	Derivative<F1> m_df1;
	typedef typename Derivative<F1>::Type Type;
	Type expression() const
	{
		return m_df1.expression();
	}
};

template <class F2>
class Derivative< Add<Const, F2> > {
public:
	Derivative< Add<Const, F2> > (const Add<Const, F2>& f)
		: m_df2(f.m_f2)
	{
	}
	double operator()(double x) const
	{
		return m_df2(x);
	}
	Derivative<F2> m_df2;
	typedef typename Derivative<F2>::Type Type;
	Type expression() const
	{
		return m_df2.expression();
	}
};

template <>
class Derivative< Add<Const, Const> > {
public:
	Derivative< Add<Const, Const> > (const Add<Const, Const>& /*f*/)
	{
	}
	double operator()(double /*x*/) const
	{
		return 0;
	}
	typedef Const Type;
	Type expression() const
	{
		return Const(0);
	}
};

template <>
class Derivative< Add<Simple, Simple> > {
public:
	Derivative< Add<Simple, Simple> > (const Add<Simple, Simple>& /*f*/)
	{
	}
	double operator()(double /*x*/) const
	{
		return 2;
	}
	typedef Const Type;
	Type expression() const
	{
		return Const(2);
	}
};

template <class F1, class F2>
class Derivative< Subtract<F1, F2> > {
public:
	Derivative< Subtract<F1, F2> > (const Subtract<F1, F2>& f)
		: m_df1(f.m_f1), m_df2(f.m_f2)
	{
	}
	double operator()(double x) const
	{
		return m_df1(x) - m_df2(x);
	}
	Derivative<F1> m_df1;
	Derivative<F2> m_df2;
	typedef Subtract<typename Derivative<F1>::Type, typename Derivative<F2>::Type> Type;
	Type expression() const
	{
		return m_df1.expression() - m_df2.expression();
	}
};

template <class F1>
class Derivative< Subtract<F1, Const> > {
public:
	Derivative< Subtract<F1, Const> > (const Subtract<F1, Const>& f)
		: m_df1(f.m_f1)
	{
	}
	double operator()(double x) const
	{
		return m_df1(x);
	}
	Derivative<F1> m_df1;
	typedef typename Derivative<F1>::Type Type;
	Type expression() const
	{
		return m_df1.expression();
	}
};

template <class F2>
class Derivative< Subtract<Const, F2> > {
public:
	Derivative< Subtract<Const, F2> > (const Subtract<Const, F2>& f)
		: m_df2(f.m_f2)
	{
	}
	double operator()(double x) const
	{
		return -m_df2(x);
	}
	Derivative<F2> m_df2;
	typedef Multiply<Const, typename Derivative<F2>::Type> Type;
	Type expression() const
	{
		return Const(-1) * m_df2.expression();
	}
};

template <>
class Derivative< Subtract<Const, Const> > {
public:
	Derivative< Subtract<Const, Const> > (const Subtract<Const, Const>& /*f*/)
	{
	}
	double operator()(double /*x*/) const
	{
		return 0;
	}
	typedef Const Type;
	Type expression() const
	{
		return Const(0);
	}
};

template <>
class Derivative< Subtract<Simple, Simple> > {
public:
	Derivative< Subtract<Simple, Simple> > (const Subtract<Simple, Simple>& /*f*/)
	{
	}
	double operator()(double /*x*/) const
	{
		return 0;
	}
	typedef Const Type;
	Type expression() const
	{
		return Const(0);
	}
};

template <class F1, class F2>
class Derivative< Multiply<F1, F2> > {
public:
	Derivative< Multiply<F1, F2> > (const Multiply<F1, F2>& f)
		: m_f1(f.m_f1), m_f2(f.m_f2), m_df1(f.m_f1), m_df2(f.m_f2)
	{
	}
	double operator()(double x) const
	{
		return m_df1(x) * m_f2(x) + m_f1(x) * m_df2(x);
	}
	F1 m_f1;
	F2 m_f2;
	Derivative<F1> m_df1;
	Derivative<F2> m_df2;
	typedef Add<Multiply<typename Derivative<F1>::Type, F2>, Multiply<F1, typename Derivative<F2>::Type> > Type;
	Type expression() const
	{
		return m_df1.expression() * m_f2 + m_f1 * m_df2.expression();
	}
};

template <class F1>
class Derivative< Multiply<F1, Const> > {
public:
	Derivative< Multiply<F1, Const> > (const Multiply<F1, Const>& f)
		: m_df1(f.m_f1), m_f2(f.m_f2)
	{
	}
	double operator()(double x) const
	{
		return m_df1(x) * m_f2.m_const;
	}
	Derivative<F1> m_df1;
	Const m_f2;
	typedef Multiply<Const, typename Derivative<F1>::Type> Type;
	Type expression() const
	{
		return m_f2 * m_df1.expression();
	}
};

template <class F2>
class Derivative< Multiply<Const, F2> > {
public:
	Derivative< Multiply<Const, F2> > (const Multiply<Const, F2>& f)
		: m_f1(f.m_f1), m_df2(f.m_f2)
	{
	}
	double operator()(double x) const
	{
		return m_f1.m_const * m_df2(x);
	}
	Const m_f1;
	Derivative<F2> m_df2;
	typedef Multiply<Const, typename Derivative<F2>::Type> Type;
	Type expression() const
	{
		return m_f1 * m_df2.expression();
	}
};

template <>
class Derivative< Multiply<Const, Const> > {
public:
	Derivative< Multiply<Const, Const> > (const Multiply<Const, Const>& /*f*/)
	{
	}
	double operator()(double /*x*/) const
	{
		return 0;
	}
	typedef Const Type;
	Type expression() const
	{
		return Const(0);
	}
};

template <class F>
class Derivative< Power<F> > {
public:
	Derivative< Power<F> > (const Power<F>& f)
		: m_f(f.m_f), m_n(f.m_n), m_df(f.m_f)
	{
	}
	double operator()(double x) const
	{
		return m_n * pow(m_f(x), m_n - 1) * m_df(x);
	}
	F m_f;
	double m_n;
	Derivative<F> m_df;
	typedef Multiply<Multiply<Const, Power<F> >, typename Derivative<F>::Type> Type;
	Type expression() const
	{
		return (Const(m_n) * Pow(m_f, m_n - 1)) * m_df.expression();
	}
};

template <>
class Derivative< Power<Const> > {
public:
	Derivative< Power<Const> > (const Power<Const>& /*f*/)
	{
	}
	double operator()(double /*x*/) const
	{
		return 0;
	}
	typedef Const Type;
	Type expression() const
	{
		return Const(0);
	}
};

template <>
class Derivative< Power<Simple> > {
public:
	Derivative< Power<Simple> > (const Power<Simple>& f)
		: m_n(f.m_n)
	{
	}
	double operator()(double x) const
	{
		return m_n * pow(x, m_n - 1);
	}
	double m_n;
	typedef Multiply<Const, Power<Simple> > Type;
	Type expression() const
	{
		return Const(m_n) * Pow(X, m_n - 1);
	}
};

template <class F1, class F2>
class Derivative< Divide<F1, F2> > {
public:
	Derivative< Divide<F1, F2> > (const Divide<F1, F2>& f)
		: m_f1(f.m_f1), m_f2(f.m_f2), m_df1(f.m_f1), m_df2(f.m_f2)
	{
	}
	double operator()(double x) const
	{
		double f2 = m_f2(x);
		return (m_df1(x) * f2 - m_f1(x) * m_df2(x)) / (f2 * f2);
	}
	F1 m_f1;
	F2 m_f2;
	Derivative<F1> m_df1;
	Derivative<F2> m_df2;
	typedef Multiply<Subtract<Multiply<typename Derivative<F1>::Type, F2>, Multiply<F1, typename Derivative<F2>::Type> >, Power<F2> > Type;
	Type expression() const
	{
		return (m_df1.expression() * m_f2 - m_f1 * m_df2.expression()) * Pow(m_f2, -2);
	}
};

template <class F2>
class Derivative< Divide<Const, F2> > {
public:
	Derivative< Divide<Const, F2> > (const Divide<Const, F2>& f)
		: m_f1(f.m_f1), m_f2(f.m_f2), m_df2(f.m_f2)
	{
	}
	double operator()(double x) const
	{
		double f2 = m_f2(x);
		return -m_f1.m_const * m_df2(x) / (f2 * f2);
	}
	Const m_f1;
	F2 m_f2;
	Derivative<F2> m_df2;
	typedef Multiply<Const, Multiply<typename Derivative<F2>::Type, Power<F2> > > Type;
	Type expression() const
	{
		return Const(-m_f1.m_const) * (m_df2.expression() * Pow(m_f2, -2));
	}
};

template <class F1>
class Derivative< Divide<F1, Const> > {
public:
	Derivative< Divide<F1, Const> > (const Divide<F1, Const>& f)
		: m_f2(1 / f.m_f2.m_const), m_df1(f.m_f1)
	{
	}
	double operator()(double x) const
	{
		return m_f2.m_const * m_df1(x);
	}
	Const m_f2;
	Derivative<F1> m_df1;
	typedef Multiply<Const, typename Derivative<F1>::Type> Type;
	Type expression() const
	{
		return m_f2 * m_df1.expression();
	}
};

template <>
class Derivative< Divide<Const, Const> > {
public:
	Derivative< Divide<Const, Const> > (const Divide<Const, Const>& f)
		: m_const(f.m_f1.m_const / f.m_f2.m_const)
	{
	}
	double operator()(double /*x*/) const
	{
		return 0;
	}
	Const m_const;
	typedef Const Type;
	Type expression() const
	{
		return Const(0);
	}
};

template <class F>
class Derivative< Exponent<F> > {
public:
	Derivative< Exponent<F> >(const Exponent<F>& f)
		: m_base(f.m_base), m_factor(log(f.m_base))
		, m_f(f.m_f), m_df(f.m_f)
	{
	}
	double operator()(double x) const
	{
		return pow(m_base, m_f(x)) * m_factor * m_df(x);
	}
	double m_base, m_factor;
	F m_f;
	Derivative<F> m_df;
	typedef Multiply<Multiply<Exponent<F>, Const>, typename Derivative<F>::Type> Type;
	Type expression() const
	{
		return Exp(m_base, m_f) * m_factor * m_df.expression();
	}
};

template <class F>
class Derivative< Logarithm<F> > {
public:
	Derivative< Logarithm<F> >(const Logarithm<F>& f)
		: m_base(f.m_base), m_factor(1 / log(m_base)), m_f(f.m_f), m_df(f.m_f)
	{
	}
	double operator()(double x) const
	{
		return m_factor * m_df(x) / x;
	}
	double m_base, m_factor;
	F m_f;
	Derivative<F> m_df;
	typedef Divide<Multiply< Const, typename Derivative<F>::Type >, Simple> Type;
	Type expression() const
	{
		return m_factor * m_df.expression() / X;
	}
};

template <class F>
class Derivative< Sine<F> > {
public:
	Derivative< Sine<F> >(const Sine<F>& f)
		: m_f(f.m_f), m_df(f.m_f)
	{
	}
	double operator()(double x) const
	{
		return cos(m_f(x)) * m_df(x);
	}
	F m_f;
	Derivative<F> m_df;
	typedef Multiply<Cosine<F>, typename Derivative<F>::Type> Type;
	Type expression() const
	{
		return Cos(m_f) * m_df.expression();
	}
};

template <class F>
class Derivative< Cosine<F> > {
public:
	Derivative< Cosine<F> >(const Cosine<F>& f)
		: m_f(f.m_f), m_df(f.m_f)
	{
	}
	double operator()(double x) const
	{
		return -sin(m_f(x)) * m_df(x);
	}
	F m_f;
	Derivative<F> m_df;
	typedef Multiply<Multiply< Const, Sine<F> >, typename Derivative<F>::Type> Type;
	Type expression() const
	{
		return (Const(-1) * Sin(m_f)) * m_df.expression();
	}
};

template <class F>
class Derivative< Tangent<F> > {
public:
	Derivative< Tangent<F> >(const Tangent<F>& f)
		: m_f(f.m_f), m_df(f.m_f)
	{
	}
	double operator()(double x) const
	{
		double cosfx = cos(m_f(x));
		return m_df(x) / (cosfx * cosfx);
	}
	F m_f;
	Derivative<F> m_df;
	typedef Divide<typename Derivative<F>::Type, Multiply< Cosine<F>, Cosine<F> > > Type;
	Type expression() const
	{
		return m_df.expression() / (Cos(m_f) * Cos(m_f));
	}
};

template <class F>
class Derivative< Cotangent<F> > {
public:
	Derivative< Cotangent<F> >(const Cotangent<F>& f)
		: m_f(f.m_f), m_df(f.m_f)
	{
	}
	double operator()(double x) const
	{
		double sinfx = sin(m_f(x));
		return -m_df(x) / (sinfx * sinfx);
	}
	F m_f;
	Derivative<F> m_df;
	typedef Divide<Multiply<Const, typename Derivative<F>::Type>, Multiply< Sine<F>, Sine<F> > > Type;
	Type expression() const
	{
		return Const(-1) * m_df.expression() / (Sin(m_f) * Sin(m_f));
	}
};

template <class F>
class Derivative< Arcsine<F> > {
public:
	Derivative< Arcsine<F> >(const Arcsine<F>& f)
		: m_f(f.m_f), m_df(f.m_f)
	{
	}
	double operator()(double x) const
	{
		double fx = m_f(x);
		return m_df(x) / sqrt(1 - fx * fx);
	}
	F m_f;
	Derivative<F> m_df;
	typedef Divide<typename Derivative<F>::Type, Power<Subtract<Const, Multiply<F, F> > > > Type;
	Type expression() const
	{
		return m_df.expression() / Sqrt(1 - m_f * m_f);
	}
};

template <class F>
class Derivative< Arccosine<F> > {
public:
	Derivative< Arccosine<F> >(const Arccosine<F>& f)
		: m_f(f.m_f), m_df(f.m_f)
	{
	}
	double operator()(double x) const
	{
		double fx = m_f(x);
		return -m_df(x) / sqrt(1 - fx * fx);
	}
	F m_f;
	Derivative<F> m_df;
	typedef Divide<Multiply<Const, typename Derivative<F>::Type>, Power<Subtract<Const, Multiply<F, F> > > > Type;
	Type expression() const
	{
		return (Const(-1) * m_df.expression()) / Sqrt(1 - m_f * m_f);
	}
};

template <class F>
class Derivative< Arctangent<F> > {
public:
	Derivative< Arctangent<F> >(const Arctangent<F>& f)
		: m_f(f.m_f), m_df(f.m_f)
	{
	}
	double operator()(double x) const
	{
		double fx = m_f(x);
		return m_df(x) / (1 + fx * fx);
	}
	F m_f;
	Derivative<F> m_df;
	typedef Divide<typename Derivative<F>::Type, Add<Const, Multiply<F, F> > > Type;
	Type expression() const
	{
		return m_df.expression() / (Const(1) + Sqr(m_f));
	}
};

template <class F>
class Derivative< Arccotangent<F> > {
public:
	Derivative< Arccotangent<F> >(const Arccotangent<F>& f)
		: m_f(f.m_f), m_df(f.m_f)
	{
	}
	double operator()(double x) const
	{
		double fx = m_f(x);
		return -m_df(x) / (1 + fx * fx);
	}
	F m_f;
	Derivative<F> m_df;
	typedef Divide<Multiply<Const, typename Derivative<F>::Type>, Add<Const, Multiply<F, F> > > Type;
	Type expression() const
	{
		return (Const(-1) * m_df.expression()) / (Const(1) + Sqr(m_f));
	}
};

template <class F>
typename Derivative<F>::Type derivative(F f)
{
	return Derivative<F>(f).expression();
}

}

#endif // CrazyMath_h
