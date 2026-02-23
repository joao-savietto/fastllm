from typing import Any, Dict

from pydantic import BaseModel, Field
from sympy.parsing.latex import parse_latex

from fastllm import tool


class MathExpression(BaseModel):
    expression: str = Field(
        ...,
        description="The mathematical expression in LaTeX format to solve or simplify (e.g., '2+2', '\\int x dx', 'x^2 + 2x + 1 = 0')",
    )


@tool(
    description="Solves or simplifies a mathematical expression provided in LaTeX format. Returns the result as a string.",
    pydantic_model=MathExpression,
)
def solve_math(request: MathExpression) -> Dict[str, Any]:
    """
    Parses a LaTeX math expression and evaluates/solves it using SymPy.
    """
    try:
        # Convert LaTeX string to SymPy object
        expr = parse_latex(request.expression)

        # Attempt to simplify and evaluate
        # .doit() evaluates derivatives, integrals, limits, etc.
        result = expr.doit()

        # Try to simplify the result further
        # Note: simplify() can be computationally expensive for very complex expressions
        simplified_result = result.simplify()

        # If the result is a number, we can also provide a float approximation
        numerical_value = None
        try:
            # Check if it can be evaluated to a number
            evalf_result = simplified_result.evalf()
            # If evalf returns a Float or can be cast to float, do it
            if evalf_result.is_number:
                numerical_value = float(evalf_result)
        except Exception:
            pass

        return {
            "original_expression": request.expression,
            "result": str(simplified_result),
            "numerical_value": numerical_value,
            "sympy_repr": str(expr),
        }

    except Exception as e:
        return {"error": f"Failed to solve expression: {str(e)}"}
