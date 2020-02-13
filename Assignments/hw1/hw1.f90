program hw1
use smoothing
implicit none

!$use OMP_LIB

!*** Number of array elements in one direction
integer :: n = 2**14,nbx,nby,istat

!*** Smothing constants
real :: a = 0.05, b = 0.1, c = 0.4

!*** Threshold
real :: t = 0.1

!*** Input and output array
real, dimension(:,:), allocatable :: x, y

!*** Allocate input array
allocate(x(0:n+1,0:n+1), stat=istat)
allocate(y(0:n+1,0:n+1), stat=istat)

!$omp parallel

write (0,*) "This is thread", omp_get_thread_num()

!$omp end parallel

!*** Initialize array x
call initialize(x, n)

!*** Derive second array from first array
call smooth(y, x, n, a, b, c)

!*** Count elements in first array
call count(x, n, t, nbx)

!*** Count elements in second array
call count(y, n, t, nby)

!*** Print number of elements below threshold
write (0,*)
write (0,'(a)')    'Summary'
write (0,'(a)')   '-------'
write (0,'(a,i14)') 'Number of elements in a row/column ::',n+2
write (0,'(a,i14)') 'Number of inner elements in a row/column :: ', n
write (0,'(a,i14)') 'Total number of elements :: ', (n+2)**2
write (0,'(a,i14)') 'Total number of inner elements :: ', n**2
write (0,'(a,f14.5)') 'Memory (GB) used per array :: ', real((n+2))**2 * 4. / (1024.)
write (0,'(a,f14.2)') 'Threshold :: ', t
write (0,'(a,3(f4.2,1x))') 'Smoothing constants (a, b, c) :: ', a, b, c
write (0,'(a,i14)') 'Number of elements below threshold (X) :: ', nbx
write (0,'(a,es14.5)') 'Fraction of elements below threshold :: ', real(nbx) / n**2
write (0,'(a,i14)') 'Number of elements below threshold (Y) :: ', nby
write (0,'(a,es14.5)') 'Fraction of elements below threshold :: ', real(nby) / n**2

    
end program hw1

