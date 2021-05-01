import sys
import codecs
import re

emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002500-\U00002BEF"  # chinese char
                           u"\U00002702-\U000027B0"
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           u"\U0001f926-\U0001f937"
                           u"\U00010000-\U0010ffff"
                           u"\u2640-\u2642"
                           u"\u2600-\u2B55"
                           u"\u200d"
                           u"\u23cf"
                           u"\u23e9"
                           u"\u231a"
                           u"\ufe0f"  # dingbats
                           u"\u3030"
                           u"\\u2680-\\u277F"
                           "]+", flags=re.UNICODE)

test_str = '\\xe2\\x9d\\xa4 RATU \\xe2\\x9d\\xa4 MAYCREATE MOISTURIZING SPRAY - LOTION SPRAY MAY CREATE 150ML ORIGINAL'
# new_test_str = r'{}'.format(test_str)
# new_test_str = repr(test_str)
# # print(test_str.decode('string_escape'))
# tmp = codecs.decode(new_test_str, 'unicode_escape')

tmp = emoji_pattern.sub(r'', test_str)

print()
