import os
os.environ['JAVAHOME'] = 'C:/Program Files/Java/jdk1.8.0_121/bin/java.exe'

# Stanford Parser [Very useful].
from nltk.parse.stanford import StanfordParser
stanford_parser_dir = 'C:/Users/AlexPC/AppData/Roaming/nltk_data/stanford-parser-full-2017-06-09/'
my_path_to_models_jar = stanford_parser_dir+ "stanford-parser-3.8.0-models.jar"
my_path_to_jar = stanford_parser_dir + "stanford-parser.jar"

sent = "this is the english parser test"

# Stanford Parser call.
english_parser = StanfordParser(my_path_to_models_jar,my_path_to_jar)
parsed_sentences = english_parser.raw_parse(sent)

# GUI
for line in parsed_sentences:
       print(line)
       line.draw()
