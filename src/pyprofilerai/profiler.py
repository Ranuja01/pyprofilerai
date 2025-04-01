"""
@author: Ranuja Pinnaduwage

This file is part of pyprofilerai, a Python package that combines profiling 
with AI-based performance optimization suggestions.

Description:
This file implements the core functionality of the pyprofilerai package, which 
provides profiling of Python code and suggests optimizations based on AI analysis.

Based on Python's cProfile module for performance analysis and utilizes Google Gemini.  
Copyright (C) 2025 Ranuja Pinnaduwage  
Licensed under the MIT License.  

Permission is hereby granted, free of charge, to any person obtaining a copy  
of this software and associated documentation files (the "Software"), to deal  
in the Software without restriction, including without limitation the rights  
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell  
copies of the Software, and to permit persons to whom the Software is  
furnished to do so, subject to the following conditions:  

The above copyright notice and this permission notice shall be included in all  
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR  
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,  
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE  
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER  
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  
SOFTWARE.
"""

import inspect
import cProfile
import pstats
import io
import os
from google import genai


def get_function_code(func):
    
    """
    Function to return the source code of a given function as text

    Parameters:
    - func: The function to be converted to text
    
    Returns:
    - String representing the given function's code
    """
    
    return inspect.getsource(func)

def get_api_key():

    """
    Function to acquire the necessary API key for Gemini

    Returns:
    - The Api key if found, else throws an error
    """
    
    api_key = os.getenv('GEMINI_API_KEY')
    if api_key is None:
        raise ValueError("API key is missing. Please set the API_KEY environment variable.")
    return api_key


def profile_code(func, *args, **kwargs):

    """
    Profiles a given function and returns the stats as a string

    Parameters:
    - func: The function to be profiled
    - *args: Possible arguments to be passed to the given function
    - func: Possible keyword arguments to be passed to the given function    

    Returns:
    - The Api key if found, else throws an error
    """
    
    # Run the profiler
    profiler = cProfile.Profile()
    profiler.enable()
    func(*args, **kwargs)
    profiler.disable()

    # Acquire and organize the results    
    result = io.StringIO()
    stats = pstats.Stats(profiler, stream=result)
    stats.strip_dirs().sort_stats(pstats.SortKey.CUMULATIVE)
    stats.print_stats()
    
    return result.getvalue()

def analyze_performance(func, *args, **kwargs):
    
    """
    Calls profiler to get performance report and acquires feedback from Gemini

    Parameters:
    - func: The function to be profiled
    - *args: Possible arguments to be passed to the given function
    - func: Possible keyword arguments to be passed to the given function    

    Returns:
    - None. The function prints the report and writes it to a file
    """
    
    # Call profiler function
    profiler_output = profile_code(func, *args, **kwargs)
    
    # Call Gemini to get suggestions
    suggestions = get_optimization_suggestions(profiler_output, get_function_code(func))

    # File path for output 
    output_file = "performance_analysis.txt"

    # Open the file in write mode (this overwrites the content on each run)
    with open(output_file, 'w') as file:
        # Write profiling results and suggestions to the file
        file.write("=== Profiling Results ===\n")
        file.write(profiler_output)
        file.write("\n=== AI Suggestions ===\n")
        file.write(suggestions)

    # Print results to the console
    print("\n=== Profiling Results ===")
    print(profiler_output)

    print("\n=== AI Suggestions ===")
    print(suggestions)

    print(f"\nResults and suggestions written to {output_file}")

def get_optimization_suggestions(profiler_output, func_code):
    
    """
    Calls Gemini with the given performance report to acquire suggestions

    Parameters:
    - func: The function to be profiled
    - *args: Possible arguments to be passed to the given function
    - func: Possible keyword arguments to be passed to the given function    

    Returns:
    - None. The function prints the report and writes it to a file
    """
    # Creates client with Api key
    client = genai.Client(api_key=get_api_key())
    
    # Creates message to send to Gemini
    content = """Given the following code and cProfile output,
                make some suggestions for improving efficiency:
                
            """
            
    content += func_code
    content += profiler_output
    
    # Acquires and returns the response
    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=content
    )
    return response.text
