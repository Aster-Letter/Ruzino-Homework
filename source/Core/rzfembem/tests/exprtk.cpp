
#include "exprtk/exprtk.hpp"

#include <iostream>
#include <string>

int main()
{
    typedef exprtk::symbol_table<double> symbol_table_t;
    typedef exprtk::expression<double> expression_t;
    typedef exprtk::parser<double> parser_t;

    const std::string expression_string = "3 * x + 2 * y - cos(z)";

    double x = 1.0;
    double y = 2.0;
    double z = 3.14159;

    symbol_table_t symbol_table;
    symbol_table.add_variable("x", x);
    symbol_table.add_variable("y", y);
    symbol_table.add_variable("z", z);
    symbol_table.add_constants();

    expression_t expression;
    expression.register_symbol_table(symbol_table);

    parser_t parser;
    parser.compile(expression_string, expression);

    double result = expression.value();

    std::cout << "Expression: " << expression_string << std::endl;
    std::cout << "x = " << x << ", y = " << y << ", z = " << z << std::endl;
    std::cout << "Result: " << result << std::endl;

    return 0;
}
