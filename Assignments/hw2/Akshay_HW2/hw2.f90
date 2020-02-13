program hw2
use mpi
use smoothing
implicit none

!*** Number of array elements in one direction 32768
integer :: n =  16386,nbx,nby,istat,nbxtot,nbytot,i

!*** Smothing constants
real :: a = 0.05, b = 0.1, c = 0.4

!*** Threshold
real :: t = 0.1

!*** For timer
double precision :: tim_st,tim_end

!*** Input and output array
real, dimension(:,:), allocatable :: x, y,z

!*** For mpi
integer :: npes,mype,ierr,masterid=0

call mpi_init(ierr)
call mpi_comm_rank(MPI_COMM_WORLD,mype,ierr)
call mpi_comm_size(MPI_COMM_WORLD,npes,ierr)

print *, 'This is thread', mype


!*** Allocate input x array
call mpi_barrier(MPI_COMM_WORLD,ierr)
tim_st=MPI_WTIME()

allocate(x(0:n+1,0:n+1), stat=istat)

call mpi_barrier(MPI_COMM_WORLD,ierr)
tim_end=MPI_WTIME()

if(mype==masterid) then
   print *, "The time taken to allocate x is" , (tim_end-tim_st)
endif

!*** Allocate output y array
call mpi_barrier(MPI_COMM_WORLD,ierr)
tim_st=MPI_WTIME()

allocate(y(0:n+1,0:n+1), stat=istat)

call mpi_barrier(MPI_COMM_WORLD,ierr)
tim_end=MPI_WTIME()

if(mype==masterid) then
    print *, "The time taken to allocate y is" , (tim_end-tim_st)
endif


!*** Initialize array x

call mpi_barrier(MPI_COMM_WORLD,ierr)
tim_st=MPI_WTIME()

if(mype==masterid) then
   call initialize(x,n)
endif

call mpi_barrier(MPI_COMM_WORLD,ierr)
tim_end=MPI_WTIME()

if(mype==masterid) then
    print *, "The time taken to initialize x is" , (tim_end-tim_st)
endif

! Broadcastin x to all procs so that it can use it for smoothing 
i=size(x)
call MPI_Bcast(x,i,MPI_REAL,masterid,MPI_COMM_WORLD,ierr)



!*** Derive second array from first array
call mpi_barrier(MPI_COMM_WORLD,ierr)
tim_st=MPI_WTIME()

call smooth(y, x, n, a, b, c,mype,npes)

call mpi_barrier(MPI_COMM_WORLD,ierr)
tim_end=MPI_WTIME()

if(mype==masterid) then
     print *, "The time taken to smooth y is" , (tim_end-tim_st)
endif


!*** Count elements in first array

call mpi_barrier(MPI_COMM_WORLD,ierr)
tim_st=MPI_WTIME()

call count(x, n, t, nbx,mype,npes)

call mpi_barrier(MPI_COMM_WORLD,ierr)
tim_end=MPI_WTIME()

call MPI_reduce(nbx,nbxtot,1,MPI_INT,MPI_SUM,masterid,MPI_COMM_WORLD,ierr)

if(mype==masterid) then
    print *, "The time taken to count x is" , (tim_end-tim_st)! real(i2-i1)/real(j)
endif


!*** Count elements in second array

call mpi_barrier(MPI_COMM_WORLD,ierr)
tim_st=MPI_WTIME()

call count(y, n, t, nby,mype,npes)

call mpi_barrier(MPI_COMM_WORLD,ierr)
tim_end=MPI_WTIME()

call MPI_reduce(nby,nbytot,1,MPI_INT,MPI_SUM,masterid,MPI_COMM_WORLD,ierr)

if(mype==masterid) then
    print *, "The time taken to count y is" , (tim_end-tim_st)! real(i2-i1)/real(j)
endif

deallocate(x, stat=istat)
deallocate(y, stat=istat)

if(mype==masterid) then
   !*** Print number of elements below threshold
   write (0,*)
   write (0,'(a)')    'Summary'
   write (0,'(a)')   '-------'
   write (0,'(a,i14)') 'Number of tasks used ::',npes
   write (0,'(a,i14)') 'Number of elements in a row/column ::',n+2
   write (0,'(a,i14)') 'Number of inner elements in a row/column :: ', n
   write (0,'(a,i14)') 'Total number of elements :: ', (n+2)**2
   write (0,'(a,i14)') 'Total number of inner elements :: ', n**2
   write (0,'(a,f14.5)') 'Memory (GB) used per array :: ', real((n+2))**2 * 4. / (1024.)
   write (0,'(a,f14.2)') 'Threshold :: ', t
   write (0,'(a,3(f4.2,1x))') 'Smoothing constants (a, b, c) :: ', a, b, c
   write (0,'(a,i14)') 'Number of elements below threshold (X) :: ', nbxtot
   write (0,'(a,es14.5)') 'Fraction of elements below threshold :: ', real(nbxtot) / n**2
   write (0,'(a,i14)') 'Number of elements below threshold (Y) :: ', nbytot
   write (0,'(a,es14.5)') 'Fraction of elements below threshold :: ', real(nbytot) / n**2
endif

call mpi_finalize(ierr)    
end program hw2

