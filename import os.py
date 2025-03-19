import os
import re

def remove_russian_comments(directory):
    
        for file in files:
                path = os.path.join(root, file)
                with open(path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                with open(path, "w", encoding="utf-8") as f:
                    for line in lines:
                            f.write(line)

remove_russian_comments(".")  