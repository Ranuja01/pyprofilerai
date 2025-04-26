# PyProfilerAI
## About This Project  
PyProfilerAI is a code profiling tool that provides AI-driven insights for optimizing code performance. This package is designed to offer a structured approach to integrating AI into software development, assisting both new and experienced developers in writing more efficient code. While large language models (LLMs) like ChatGPT have become popular tools for code improvement, their suggestions often lack real performance insights, leading to unintended inefficiencies and/or bugs.

To address this, PyProfilerAI combines Python's built-in [`cProfile`](https://docs.python.org/3/library/profile.html) with Google's Gemini AI to analyze function call performance and generate informed optimization suggestions. Unlike the common approach of simply pasting code into an LLM and asking for improvements, this tool provides the model with real execution data, allowing for more precise and context-aware recommendations. By considering actual runtime performance and function calls including those in external modules PyProfilerAI delivers smarter, data-driven optimizations rather than speculative advice. *Note:* the suggestions are AI generated and therefore extra caution should be taken before implementing them, especially for more sensitive projects.

## Build Requirements
Ensure you have Python 3.8 or greater installed on your system along with an udpated version of pip. *Note:* you must acquire a Gemini API key; they are free to use but you should use them at your own discretion. If you do not wish to input your code with an LLM, then you should look for other alternatives. The following are the instructions for how you can acquire an API key and how to set your environment variables for this package to work:

- Go to: https://ai.google.dev/gemini-api/docs
- Click on Get a Gemini API key.
- Click on Create API key.
- Once you've created your key, make sure to save it in a safe place, you can't find it again.
- Now go into your system's environment variables and add a variable called GEMINI_API_KEY and for the value paste your API key.
- If you already have a terminal or IDE open, you may have to restart your session so that the environment can see the updated variables.

## Download
To download this package, use the following instructions:
  ```sh
  pip install pyprofilerai
  ```
If this does not work, then simply:
  ```sh
  pip install git+https://github.com/Ranuja01/pyprofilerai.git
  ```
## Example Usage
To use the module, simply import pyprofilerai as seen below:
```python
import pyprofilerai
```
This package currently only works with functions. Meaning that the code that you wish to profile must be enclosed within a function that you pass as seen in the examples below:
```python
import pyprofilerai
def test_function(num, exp):
    for i in range(num):
        i**exp
    
pyprofilerai.analyze_performance(test_function,100,2)
```
The above example is relatively simple, yet a comprehensive report is created to guide the user. The output for this function call can be seen below. *Note:* the formatting looks much better on the actual textfile that is printed:

### Output
=== Profiling Results ===
         2 function calls in 0.000 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function<br>
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects<br>
        1    0.000    0.000    0.000    0.000 example_usages.py.py:10(test_function)



=== AI Suggestions ===
Okay, based on the provided code and `cProfile` output, here's an analysis and suggestions for improvement.

**Understanding the Situation**

*   **The Code:**

    ```python
    def test_function(num, exp):
        for i in range(num):
            i**exp
    ```

    The `test_function` calculates `i` raised to the power of `exp` in a loop. The result isn't stored or used.

*   **The `cProfile` Output:**

    *   `2 function calls in 0.000 seconds`:  Very little time is being spent.  The numbers are essentially zero.
    *   `ncalls`: Number of calls.
    *   `tottime`: Total time spent in the function (excluding calls to sub-functions).
    *   `percall`:  `tottime` divided by `ncalls`.
    *   `cumtime`: Cumulative time spent in the function (including calls to sub-functions).
    *   `percall`: `cumtime` divided by primitive calls.
    *   The output shows that the entire execution takes a negligible amount of time. The profiler barely registers any activity.

**Analysis and Suggestions**

Because the profiler barely registers any activity, it's difficult to offer meaningful optimization advice. The following are some possibilities, but they're unlikely to make a substantial difference:

1.  **Check for unnecessary overhead**:

    In this case, the problem is that the line `i**exp` calculates a value which is never stored or used.  Simply remove the line `i**exp`, unless you have a reason to keep it there.

2.  **Consider alternatives to the `**` operator**

    The `**` operator does exponentiation using floating-point arithmetic if exp is a non-integer value.  It's possible that this overhead is substantial.

3.  **Vectorization with NumPy (if applicable and if the result *is* needed)**:

    If you actually *need* to store the results of the exponentiation, and you're going to be doing a lot of these calculations, NumPy could be beneficial:

    ```python
    import numpy as np

    def test_function_numpy(num, exp):
        i = np.arange(num)  # Create a NumPy array of i values
        result = i ** exp    # NumPy will vectorize the exponentiation
        return result
    ```

    NumPy's vectorized operations can be significantly faster than Python loops for numerical computations.

4.  **Memoization (if applicable and if `exp` and `num` stay the same)**

    If `test_function` is being called multiple times with the *same* `num` and `exp` values, memoization *could* provide a speedup by caching the results. However, given the simplicity of the operation, the overhead of memoization might outweigh any benefits unless it is being called with the same arguments very many times.

    ```python
    from functools import lru_cache

    @lru_cache(maxsize=None)  # Cache all calls
    def test_function(num, exp):
        result = []
        for i in range(num):
            result.append(i**exp) # store i**exp in a list
        return result
    ```

**Important Considerations**

*   **Is the Code Representative?** The `cProfile` output suggests that `test_function` is not the bottleneck in your *actual* application.  The profiler is telling you that the function is not taking much time at all, so focus your optimization efforts elsewhere.  Make sure the `test_function` you are profiling is truly representative of how it's used in a real-world scenario.  If you're using it in a bigger program, profile the *entire* program to find the *real* bottlenecks.
*   **Micro-optimizations:**  The above suggestions are, at best, micro-optimizations.  Don't spend a lot of time on these unless you have a very specific, measurable reason to believe they will make a difference.

In summary, the provided code is so simple that optimizing it is unlikely to provide a meaningful performance improvement. Focus on identifying and optimizing more significant bottlenecks in your larger application.  If you need to actually use the result of the calculation, NumPy might offer vectorization advantages.

### Other Examples
For a use case on a more detailed function, visit the package folder and open example_usages.py for this example as well as an example using a more complex function. The example trains a tensorflow model which is a much more time consuming task that can heavily benefit from optimizations. Run the example_usage.py file to see the results!
