import sys
paragraph = []
line_length = 64
space_size = 2

def set_line(line,tot,final):
    shortfall = line_length-tot; space = space_size*' '
    for w in range(len(line)):
        sys.stdout.write(line[w])
        sys.stdout.write(space)
        if final!=1 and w<shortfall: sys.stdout.write(' ')
    print

total_chars = -1
while 1:
    a = raw_input()
    if a == "EOF":
        set_line(paragraph,total_chars,1)
        break
    new_total = total_chars+len(a)+space_size
    if new_total>line_length:
        set_line(paragraph,total_chars,0)
        paragraph = [a]; total_chars = len(a);
    else:
        paragraph.append(a)
        total_chars = new_total
