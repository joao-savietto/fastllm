import json
import unittest

from fastllm.tools.math_solver import MathExpression, solve_math


class TestMathSolver(unittest.TestCase):
    def test_simple_addition(self):
        # 2 + 2
        expr = MathExpression(expression="2+2")
        # execute returns a JSON string
        result_json = solve_math.execute(expr)
        result = json.loads(result_json)

        self.assertEqual(result["result"], "4")
        self.assertEqual(result["numerical_value"], 4.0)

    def test_integration(self):
        # Integral of x dx -> x^2/2
        expr = MathExpression(expression="\\int x dx")
        result_json = solve_math.execute(expr)
        result = json.loads(result_json)

        # SymPy usually returns x**2/2
        self.assertEqual(result["result"], "x**2/2")

    def test_simplification(self):
        # x + x -> 2x
        expr = MathExpression(expression="x + x")
        result_json = solve_math.execute(expr)
        result = json.loads(result_json)

        self.assertEqual(result["result"], "2*x")

    def test_evalf(self):
        # sqrt(2) approx 1.414
        expr = MathExpression(expression="\\sqrt{2}")
        result_json = solve_math.execute(expr)
        result = json.loads(result_json)

        # Check if numerical_value is present and close
        self.assertIsNotNone(result.get("numerical_value"))
        self.assertAlmostEqual(
            result["numerical_value"], 1.41421356, places=5
        )

    def test_division_by_zero(self):
        # Division by zero -> zoo (complex infinity)
        expr = MathExpression(expression="\\frac{1}{0}")
        result_json = solve_math.execute(expr)
        result = json.loads(result_json)

        # Should return zoo or handle it
        self.assertIn("zoo", result["result"])


if __name__ == "__main__":
    unittest.main()
