PROGRAM test_system_clock
    INTEGER :: count, count_rate, count_max,x=0
    INTEGER :: count1, count_rate1, count_max1
    CALL SYSTEM_CLOCK(count, count_rate, count_max)
    WRITE(*,*) count, count_rate, count_max
    do i=1,10000
       x=x+2

    end do
    CALL SYSTEM_CLOCK(count1, count_rate1, count_max1)
    WRITE(*,*) count1, count_rate1, count_max1

END PROGRAM
