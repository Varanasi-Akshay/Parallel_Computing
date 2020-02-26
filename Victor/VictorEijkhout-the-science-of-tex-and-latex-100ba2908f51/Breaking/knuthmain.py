def paragraphbreak():
    active = [0]
    init_costs()
    nwords = len(paragraph)
    for w in range(1,nwords):
        # compute the cost of breaking after word w
        print "Recent word",w
        for a in active:
            line = paragraph[a:w+1]
            if w==nwords-1:
                ratio = 0 # last line will be set perfect
            else:
                ratio = compute_ratio(line)
                print "..line=",line,"; ratio=",ratio
            if ratio<-1:
                active.remove(a)
                print "active point",a,"removed"
            else:
                update_cost(a,w,ratio)
        report_cost(w)
        active.append(w)
        print
    final_report()
