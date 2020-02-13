program hw2
include 'mpif.h'
use smoothing
implicit none


!*** Number of array elements in one direction 32768
integer :: n =  16386,nbx,nby,istat

!*** Smothing constants
real :: a = 0.05, b = 0.1, c = 0.4

!*** Threshold
real :: t = 0.1

!*** For timer
integer :: i1,i2,j

!*** Input and output array
real, dimension(:,:), allocatable :: x, y

!*** For mpi
integer :: npes,mype,ierr,masterid=0

call mpi_init(ierr)
call mpi_comm_rank(MPI_COMM_WORLD,mype,ierr)
call mpi_comm_size(MPI_COMM_WORLD,npes,ierr)




!*** Allocate input array
call system_clock(i1,j)
allocate(x(0:n+1,0:n+1), stat=istat)
call system_clock(i2,j)
if(mype==masterid) then
    print *, "The time taken to allocate x is" , real(i2-i1)/real(j)

call system_clock(i1,j)
allocate(y(0:n+1,0:n+1), stat=istat)
call system_clock(i2,j)
if(mype==masterid) then
    print *, "The time taken to allocate y is" , real(i2-i1)/real(j)



print *, 'This is thread', omp_get_thread_num()



!*** Initialize array x
call system_clock(i1,j)
call initialize(x, n)
call system_clock(i2,j)
if(mype==masterid) then
    print *, "The time taken to initialize x is" , real(i2-i1)/real(j)


!*** Derive second array from first array
!! ! omp parallel
! call system_clock(i1,j)
! call smooth(y, x, n, a, b, c)
! call system_clock(i2,j)
! if(mype==masterid)
!     print *, "The time taken to smooth y is" , real(i2-i1)/real(j)

! !! omp end parallel


! !! omp parallel
! !*** Count elements in first array
! call system_clock(i1,j)
! call count(x, n, t, nbx)
! call system_clock(i2,j)
! if(mype==masterid)
!     print *, "The time taken to count x is" , real(i2-i1)/real(j)



! !*** Count elements in second array
! call system_clock(i1,j)
! call count(y, n, t, nby)
! call system_clock(i2,j)
! if(mype==masterid)
!     print *, "The time taken to count y is" , real(i2-i1)/real(j)


!! omp end parallel

if(mype==masterid) then
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

call mpi_finalize(ierr)    
end program hw2

