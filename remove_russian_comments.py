import os
import re
from googletrans import Translator

# Initialize the Google Translator
translator = Translator()

# Regular expression to match comments (Fixed pattern)
pattern = re.compile(r"#\s*(.+)")  # Matches any comment starting with #

def translate_comment(comment):
    """
    Translate a comment from Russian to English.
    
    Args:
        comment (str): The Russian comment to be translated.
    
    Returns:
        str: The translated English comment.
    """
    try:
        return translator.translate(comment, src="ru", dest="en").text
    except Exception as e:
        print(f"Translation error: {e}")
        return comment  # Return original if translation fails

def replace_russian_comments(directory):
    """
    Iterate through all Python files in the given directory and translate Russian comments to English.
    
    Args:
        directory (str): The directory path to scan for Python files.
    """
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):  # Process only Python files
                path = os.path.join(root, file)
                
                # Read the file content
                with open(path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                # Write the modified content back to the file
                with open(path, "w", encoding="utf-8") as f:
                    for line in lines:
                        match = pattern.search(line)
                        if match:
                            russian_comment = match.group(1)  # Extract the Russian comment
                            translated_comment = translate_comment(russian_comment)  # Translate it
                            line = line.replace(russian_comment, translated_comment)  # Replace it in the line
                        f.write(line)  # Write the modified line back

# Run the function on the current directory
replace_russian_comments(".")
