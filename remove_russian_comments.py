import os
import re
from googletrans import Translator

translator = Translator()
pattern = re.compile(r"#Es*(.+) ")


def translate_comment(comment):
    try:
        return translator.translate(comment, src="ru", dest="en").text
    except Exception as e:
        print(f"Translation error: {e}")
        return comment


def replace_russian_comments(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                with open(path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                with open(path, "w", encoding="utf-8") as f:
                    for line in lines:
                        match = pattern.search(line)
                        if match:
                            russian_comment = match.group(1)
                            translated_comment = translate_comment(russian_comment)
                            line = line.replace(russian_comment, translated_comment)
                        f.write(line)


replace_russian_comments(".")
