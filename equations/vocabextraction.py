import csv
import inspect
import importlib.util
import re

# Dynamically import the Python file
# This module extracts all the formula information from the srsd
file_path = 'feynman.py'  # Replace with the path to your Python file
spec = importlib.util.spec_from_file_location("module.name", file_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


# Function to extract Equation and Vars from a class docstring
def extract_info_from_docstring(docstring):
    lines = docstring.strip().split("\n")
    equation = None
    vars_names = []

    for line in lines:
        if "Equation:" in line:
            equation = line.split(":")[1].strip()
        if "- x[" in line:
            match = re.search(r'- x\[\d+\]: (.+?) \(', line)
            if match:  # Check if the search found a match
                var_name = match.group(1)
                vars_names.append(var_name)

    return equation, vars_names


def extract_info(filename):
    info_dict = {}
    equation_name = ""
    with open(filename, 'r') as file:
        lines = file.readlines()

    for line in lines:
        if 'Equation:' in line:
            equation_name = line.split('Equation:')[1].strip()
        if 'Raw:' in line and equation_name:
            raw_string = line.split('Raw:')[1].strip()
            raw_string = raw_string.replace('**2', 'sqr')  # Convert **2 to sqr
            math_symbols = re.findall(r"(\*\*|[\+\-\*/\*\^\(\)])", raw_string)
            constants = re.findall(r"\b([a-df-zA-DF-Z])\b", raw_string)
            info_dict[equation_name] = {
                'Math symbols': list(set(math_symbols)),
                'Constants': list(set(constants))
            }
            equation_name = ""  # Reset for the next equation block

    return info_dict


# Initialize CSV file
with open('equations_and_vars.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Equation', 'Vars'])

    # Loop through each class in the module and extract information
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj):
            docstring = inspect.getdoc(obj)
            if docstring:  # Only proceed if the class has a docstring
                equation, vars_names = extract_info_from_docstring(docstring)
                class_name = obj.__name__
                vars_str = ', '.join(vars_names)

                # Write to CSV
                csvwriter.writerow([equation, vars_str])

# Usage:
info_dict = extract_info(file_path)
print(info_dict)
