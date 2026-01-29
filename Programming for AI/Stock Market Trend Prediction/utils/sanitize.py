import os
import re

def sanitize_table_name(filename):
    # Remove extension
    name = os.path.splitext(filename)[0]
    # Replace non-alphanumeric characters with underscores and lowercase it
    clean_name = re.sub(r'[^a-zA-Z0-0]', '_', name).lower()
    # Ensure it starts with a letter (standard SQL requirement)
    return f"raw_{clean_name}"