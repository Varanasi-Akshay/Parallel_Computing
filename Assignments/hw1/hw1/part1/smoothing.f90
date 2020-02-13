module smoothing
use OMP_LIB
implicit none
!real :: 
contains
    !*** Initialize with random numbers
    subroutine initialize(x, n)
    
    integer :: n
    real, dimension(0:n+1,0:n+1) :: x
    call random_number(x)
	end subroutine
    
    

    !*** Smooth data
    subroutine smooth(y, x, n, a, b, c)
!    use omp_lib
    integer :: i,j,n
    real :: a,b,c
    real, dimension(0:n+1,0:n+1) :: x, y
    !!!$omp parallel
    !$omp parallel do schedule(static) 
       do j=1, n
	    do i=1, n
               y(i,j) = a * (x(i-1,j-1) + x(i-1,j+1) + &
                        x(i+1,j-1) + x(i+1,j+1)) + &
                        b * (x(i-0,j-1) + x(i-0,j+1) + &
                        x(i-1,j-0) + x(i+1,j+0)) + &
                        c * x(i,j)
	    enddo
       enddo
    !!!$ omp end parallel
	end subroutine
   

    !*** Count elements below threshold
    subroutine count(x, n, t, nbx)
!    use omp_lib
    integer :: i,j,n,nbx
    real, dimension(0:n+1,0:n+1) :: x
    real :: t
    nbx = 0
    !$omp parallel do reduction(+:nbx)
      do j=1, n
        do i=1, n
           if (x(i,j) < t) then
               nbx = nbx + 1
            endif
        enddo
      enddo
    end subroutine



end module smoothing
