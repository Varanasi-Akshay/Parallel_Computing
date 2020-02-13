program main
use mpi
implicit none
integer :: rank,size,ierr,i,j
integer :: randsize
integer,allocatable,dimension(:) :: randseed
real :: random_value,max
integer :: time(8),maxproc
real,allocatable,dimension(:) :: indexfinal

call mpi_init(ierr)
call mpi_comm_rank(MPI_COMM_WORLD,rank,ierr)
call mpi_comm_size(MPI_COMM_WORLD,size,ierr)
print *,'Hi'

call mpi_barrier(MPI_COMM_WORLD,ierr)

call date_and_time(values=time)

call random_seed(size=randsize)
allocate(randseed(randsize))

allocate(indexfinal(size))

do i=1,randsize
randseed(i) = time(i)*rank
end do

call random_seed(put=randseed)
call random_number(random_value)

print *,'My rank is ',rank,'and random number calculated is', random_value 

call mpi_gather(random_value,1, MPI_REAL,indexfinal,1, MPI_REAL,0, MPI_COMM_WORLD,ierr)
call mpi_reduce(random_value,max,1,MPI_REAL,MPI_MAX,0,MPI_COMM_WORLD,ierr)

call mpi_barrier(MPI_COMM_WORLD,ierr)

if(rank==0) then
   print *, 'Total no.of procs is',size
   print *, 'Verifying if gather worked out'
   do j=1,size
      if(max==indexfinal(j)) then
         maxproc=j-1
      end if
      print *,'Value of random number in processor ',j-1,'is',indexfinal(j)
   end do
end if

call mpi_barrier(MPI_COMM_WORLD,ierr)
if(rank==0) then
   print *,'The max random value is',max,' and it is calculated by processor',maxproc
end if

call mpi_finalize(ierr)
end program main 
