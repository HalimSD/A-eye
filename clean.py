
import re



lines = []
pattern1 = re.compile('\d:\d\d')
pattern2 = re.compile('\d\d:\d\d')

with open ('./text.txt', 'r') as f:
    for line in f:
        if not re.match(pattern1, line) and not re.match(pattern2, line):
            lines.append(line)

print(len(lines))
with open('./text.txt', 'w') as f:
    f.writelines(lines)
    # re.sub(pattern, '', f)
