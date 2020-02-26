import re
if re.compile("^[0-9]+$").search("19"):
    print 'yes'
else:
    print 'no'
    
