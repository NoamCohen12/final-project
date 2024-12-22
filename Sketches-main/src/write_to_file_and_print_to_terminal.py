import os
import sys
from datetime import datetime

class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)

    def flush(self):
        for f in self.files:
            f.flush()

def save_terminal_output_to_report(func) -> None:
    # Find the current directory of the project
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # If the report directory does not exist, create it
    reports_dir = os.path.join(base_dir, "report")
    os.makedirs(reports_dir, exist_ok=True)

    # Create a file name with the current date and time
    now = datetime.now()
    timestamp = now.strftime("date_%Y-%m-%d_time_%H-%M-%S")  # YYYY-MM-DD_HH-MM-SS
    file_name = f"report_{timestamp}.txt"
    file_path = os.path.join(reports_dir, file_name)

    # Save the terminal output to the file and print it to the terminal
    with open(file_path, "w") as file:
        original_stdout = sys.stdout  # Save the original stdout
        sys.stdout = Tee(sys.stdout, file)  # Redirect the output to the file and the terminal

        try:
            # Place the code you want to run here
            func()  # Replace this with your actual function or code
        finally:
            # Restore the original stdout
            sys.stdout = original_stdout

    # Print the file saved message
    print(f"Output saved to: {file_path}")

