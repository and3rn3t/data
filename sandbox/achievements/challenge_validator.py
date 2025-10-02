"""
Challenge Auto-Validation System
Automated testing and instant feedback for challenges
"""

import ast
import traceback
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class ChallengeValidator:
    """
    Automated challenge validation with safe code execution
    and instant feedback
    """

    def __init__(self) -> None:
        self.safe_builtins = {
            # Safe built-in functions
            "abs": abs,
            "all": all,
            "any": any,
            "bin": bin,
            "bool": bool,
            "chr": chr,
            "dict": dict,
            "dir": dir,
            "enumerate": enumerate,
            "filter": filter,
            "float": float,
            "format": format,
            "frozenset": frozenset,
            "hex": hex,
            "id": id,
            "int": int,
            "isinstance": isinstance,
            "issubclass": issubclass,
            "iter": iter,
            "len": len,
            "list": list,
            "map": map,
            "max": max,
            "min": min,
            "oct": oct,
            "ord": ord,
            "pow": pow,
            "print": print,
            "range": range,
            "reversed": reversed,
            "round": round,
            "set": set,
            "slice": slice,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "type": type,
            "zip": zip,
            # Safe modules that are commonly needed
            "pd": pd,
            "pandas": pd,
            "np": np,
            "numpy": np,
        }

        self.restricted_keywords = [
            "import",
            "exec",
            "eval",
            "compile",
            "open",
            "file",
            "__import__",
            "globals",
            "locals",
            "vars",
            "dir",
            "hasattr",
            "getattr",
            "setattr",
            "delattr",
            "callable",
        ]

    def validate_challenge(
        self, challenge_id: str, user_code: str, test_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Validate user code against challenge requirements

        Args:
            challenge_id: Identifier for the challenge
            user_code: User's submitted code
            test_data: Optional test data for the challenge

        Returns:
            Validation results with score, feedback, and suggestions
        """
        result: Dict[str, Any] = {
            "challenge_id": challenge_id,
            "success": False,
            "score": 0,
            "max_score": 100,
            "feedback": [],
            "errors": [],
            "warnings": [],
            "suggestions": [],
            "execution_time": 0,
            "skills_demonstrated": [],
        }

        try:
            # Step 1: Security validation
            security_check = self._validate_security(user_code)
            if not security_check["safe"]:
                result["errors"].extend(security_check["issues"])
                result["feedback"].append(
                    "❌ Code contains potentially unsafe operations"
                )
                return result

            # Step 2: Syntax validation
            syntax_check = self._validate_syntax(user_code)
            if not syntax_check["valid"]:
                result["errors"].extend(syntax_check["errors"])
                result["feedback"].append("❌ Code has syntax errors")
                return result

            # Step 3: Execute code and run tests
            execution_result = self._execute_code_safely(user_code, test_data)
            result.update(execution_result)

            # Step 4: Analyze code quality and style
            quality_analysis = self._analyze_code_quality(user_code)
            result["warnings"].extend(quality_analysis["warnings"])
            result["suggestions"].extend(quality_analysis["suggestions"])
            result["skills_demonstrated"].extend(quality_analysis["skills"])

            # Step 5: Calculate final score
            result["score"] = self._calculate_score(result)
            result["success"] = result["score"] >= 70  # 70% threshold for success

            # Step 6: Generate personalized feedback
            result["feedback"] = self._generate_feedback(result)

        except Exception as e:
            result["errors"].append(f"Validation system error: {str(e)}")
            result["feedback"].append("❌ Unexpected error during validation")

        return result

    def _validate_security(self, code: str) -> Dict[str, Any]:
        """Check code for security issues"""
        issues = []

        # Check for restricted keywords
        for keyword in self.restricted_keywords:
            if keyword in code:
                issues.append(f"Restricted keyword '{keyword}' found in code")

        # Check for dangerous patterns
        dangerous_patterns = [
            "__",
            "exec(",
            "eval(",
            "compile(",
            "open(",
            "file(",
            "subprocess",
            "os.system",
            "os.popen",
        ]

        for pattern in dangerous_patterns:
            if pattern in code:
                issues.append(f"Potentially dangerous pattern '{pattern}' detected")

        return {"safe": len(issues) == 0, "issues": issues}

    def _validate_syntax(self, code: str) -> Dict[str, Any]:
        """Check code syntax"""
        try:
            ast.parse(code)
            return {"valid": True, "errors": []}
        except SyntaxError as e:
            return {
                "valid": False,
                "errors": [f"Syntax error at line {e.lineno}: {e.msg}"],
            }

    def _execute_code_safely(
        self, code: str, test_data: Optional[Dict]
    ) -> Dict[str, Any]:
        """Execute code in safe environment and run tests"""
        result = {
            "execution_success": False,
            "output": "",
            "test_results": [],
            "execution_time": 0,
        }

        try:
            # Create safe execution environment
            safe_globals: Dict[str, Any] = {"__builtins__": {}}
            safe_globals.update(self.safe_builtins)

            # Add test data if provided
            if test_data:
                safe_globals.update(test_data)

            # Capture output
            stdout_capture = StringIO()
            stderr_capture = StringIO()

            # Execute code
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, safe_globals)

            result["execution_success"] = True
            result["output"] = stdout_capture.getvalue()

            # Run challenge-specific tests if available
            if test_data and "tests" in test_data:
                result["test_results"] = self._run_challenge_tests(
                    safe_globals, test_data["tests"]
                )

        except Exception as e:
            result["output"] = f"Execution error: {str(e)}\n{traceback.format_exc()}"

        return result

    def _run_challenge_tests(self, environment: Dict, tests: List[Dict]) -> List[Dict]:
        """Run automated tests against user code"""
        test_results = []

        for test in tests:
            test_result = {
                "name": test.get("name", "Test"),
                "passed": False,
                "expected": test.get("expected"),
                "actual": None,
                "points": test.get("points", 10),
            }

            try:
                # Execute test
                if "assertion" in test:
                    # Simple assertion test
                    test_result["passed"] = eval(test["assertion"], environment)
                elif "function_call" in test:
                    # Function call test
                    actual = eval(test["function_call"], environment)
                    test_result["actual"] = actual
                    test_result["passed"] = actual == test["expected"]

            except Exception as e:
                test_result["error"] = str(e)

            test_results.append(test_result)

        return test_results

    def _analyze_code_quality(self, code: str) -> Dict[str, Any]:
        """Analyze code for quality, style, and skills demonstrated"""
        analysis: Dict[str, List[str]] = {
            "warnings": [],
            "suggestions": [],
            "skills": [],
        }

        try:
            tree = ast.parse(code)

            # Analyze AST for patterns
            for node in ast.walk(tree):
                # Check for pandas usage
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                    if node.value.id == "pd" and "pandas" not in analysis["skills"]:
                        analysis["skills"].append("pandas")

                # Check for numpy usage
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                    if node.value.id == "np" and "numpy" not in analysis["skills"]:
                        analysis["skills"].append("numpy")

                # Check for list comprehensions (good practice)
                if isinstance(node, ast.ListComp):
                    analysis["skills"].append("list_comprehension")

                # Check for function definitions
                if isinstance(node, ast.FunctionDef):
                    analysis["skills"].append("function_definition")

                    # Check for docstrings
                    if (
                        node.body
                        and isinstance(node.body[0], ast.Expr)
                        and isinstance(node.body[0].value, ast.Str)
                    ):
                        analysis["skills"].append("documentation")

                # Check for error handling
                if isinstance(node, ast.Try):
                    analysis["skills"].append("error_handling")

            # Code style suggestions
            lines = code.split("\n")
            for i, line in enumerate(lines, 1):
                if len(line) > 88:
                    analysis["warnings"].append(
                        f"Line {i} is longer than 88 characters"
                    )

                if (
                    line.strip()
                    and not line.startswith(" ")
                    and not line.startswith("\t")
                    and "=" in line
                    and " = " not in line
                    and "+=" not in line
                ):
                    analysis["suggestions"].append(
                        f"Line {i}: Consider adding spaces around '='"
                    )

        except Exception:
            pass  # AST analysis is optional

        return analysis

    def _calculate_score(self, result: Dict) -> int:
        """Calculate final score based on various factors"""
        score = 0

        # Base execution score
        if result.get("execution_success", False):
            score += 30

        # Test results score
        test_results = result.get("test_results", [])
        if test_results:
            passed_tests = sum(1 for test in test_results if test.get("passed", False))
            total_tests = len(test_results)
            test_score = (passed_tests / total_tests) * 50 if total_tests > 0 else 0
            score += int(test_score)

        # Code quality bonus
        skills = result.get("skills_demonstrated", [])
        quality_bonus = min(len(skills) * 2, 20)  # Up to 20 bonus points
        score += quality_bonus

        # Penalties
        warnings = len(result.get("warnings", []))
        errors = len(result.get("errors", []))
        penalty = min(warnings * 2 + errors * 5, 30)
        score -= penalty

        return max(0, min(100, int(score)))

    def _generate_feedback(self, result: Dict) -> List[str]:
        """Generate personalized feedback based on results"""
        feedback = []

        score = result.get("score", 0)

        if score >= 90:
            feedback.append("🌟 Excellent work! Your solution is outstanding.")
        elif score >= 80:
            feedback.append("✅ Great job! Your solution works well.")
        elif score >= 70:
            feedback.append("👍 Good work! Your solution meets the requirements.")
        else:
            feedback.append("💡 Keep working on it. Here are some areas to improve:")

        # Test-specific feedback
        test_results = result.get("test_results", [])
        if test_results:
            passed = sum(1 for test in test_results if test.get("passed", False))
            total = len(test_results)
            feedback.append(f"✓ Passed {passed}/{total} automated tests")

            for test in test_results:
                if not test.get("passed", False):
                    feedback.append(
                        f"❌ {test['name']}: Expected {test.get('expected')}, got {test.get('actual')}"
                    )

        # Skills feedback
        skills = result.get("skills_demonstrated", [])
        if skills:
            feedback.append(f"🎯 Skills demonstrated: {', '.join(skills)}")

        # Suggestions
        suggestions = result.get("suggestions", [])
        if suggestions:
            feedback.append("💡 Suggestions:")
            feedback.extend([f"  • {suggestion}" for suggestion in suggestions[:3]])

        return feedback

    def create_challenge_template(
        self, challenge_id: str, description: str, tests: List[Dict]
    ) -> Dict[str, Any]:
        """Create a template for a new auto-validated challenge"""
        return {
            "id": challenge_id,
            "title": challenge_id.replace("_", " ").title(),
            "description": description,
            "validation": {
                "enabled": True,
                "tests": tests,
                "time_limit": 300,  # 5 minutes
                "memory_limit": 128,  # MB
            },
            "hints": [
                "Read the problem carefully",
                "Test your code with sample data",
                "Consider edge cases",
            ],
            "skills": [],  # Will be auto-detected
            "difficulty": "beginner",
        }


class ChallengeHintSystem:
    """
    Progressive hint system for challenges
    """

    def __init__(self) -> None:
        self.hint_levels = ["general", "specific", "code_example", "solution"]

    def get_hint(
        self, challenge_id: str, _user_progress: Dict, hint_level: int = 0
    ) -> Dict[str, Any]:
        """
        Get progressive hints based on user progress

        Args:
            challenge_id: Challenge identifier
            _user_progress: User's current progress on the challenge (reserved for future use)
            hint_level: Level of hint (0=general, 1=specific, 2=code_example, 3=solution)
        """
        # Future: Load challenge-specific hints based on challenge_id
        # hints_file = Path(BASE_DIR) / "challenges" / f"{challenge_id}_hints.json"

        default_hints = {
            "general": [
                "Break down the problem into smaller steps",
                "Think about what data structures you need",
                "Consider the expected output format",
            ],
            "specific": [
                "Check the data types of your variables",
                "Make sure you're handling edge cases",
                "Verify your logic with simple examples",
            ],
            "code_example": [
                "Here's a similar pattern you can use:",
                "Consider using pandas methods like .groupby() or .apply()",
                "Remember to handle missing data appropriately",
            ],
            "solution": [
                "Here's the step-by-step approach:",
                "1. Load and examine the data",
                "2. Apply the required transformations",
                "3. Return the result in the expected format",
            ],
        }

        hint_type = self.hint_levels[min(hint_level, len(self.hint_levels) - 1)]

        return {
            "hint_level": hint_level,
            "hint_type": hint_type,
            "hints": default_hints[hint_type],
            "cost": hint_level * 5,  # XP cost for hints
            "next_available": hint_level < len(self.hint_levels) - 1,
        }
