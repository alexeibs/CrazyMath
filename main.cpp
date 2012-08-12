#include <iostream>
#include "CrazyMath.h"

using namespace std;
using namespace CrazyMath;

auto global = Tg(X) + Ctg(X) + Asin(X) * Acos(X) - Atg(X) / Actg(X);
auto d_global = derivative(global);

int main()
{
	auto f1 = (Pow(X, 3) + 2 * Sqr(X) - 4 * X + 1 / Sqrt(X)) * (Sin(X) + Cos(X) * (Log(5, X) - Exp(2, X)));
	auto f2 = derivative(f1);
	auto f3 = [](double x) -> double { return sin(x); };
	auto df1 = derivative(f1);
	auto df2 = derivative(f2);
	auto df3 = derivative(f3);
	
	cout << "f(x)\t\tf'(x)" << endl;
	cout << f1(0.5) << " \t" << df1(0.5) << endl;
	cout << f2(0.5) << " \t" << df2(0.5) << endl;
	cout << f3(0) << " \t" << df3(0) << endl;
	cout << global(0.5) << " \t" << d_global(0.5) << endl;
	
	char temp[4];
	cout << "\nPress ENTER to exit..." << endl;
	cin.getline(temp, 3);
	return 0;
}