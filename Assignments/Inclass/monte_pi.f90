program pivalue
implicit none
include 'mpif.h'
Real :: n = 10000000
integer :: i,count=0,ierr,tot_count=0,myid,numproc
Real :: x,y,z,pi,chunk



call mpi_init(ierr)
call mpi_comm_rank(MPI_COMM_WORLD,myid,ierr)
call mpi_comm_size(MPI_COMM_WORLD,numproc,ierr)
chunk=int(n/numproc)

do i=1+chunk*myid,1+chunk*(myid+1)
   call random_number(x)
   call random_number(y)
   z=sqrt((x*x)+(y*y))
   if (z<=1) then 
      count=count+1
   end if   


end do
call mpi_reduce(count,tot_count,1,mpi_int,mpi_sum,0,MPI_COMM_WORLD,ierr)
if(myid==0) then
   pi=4.0*(tot_count/n)
   print *, 'The value of pi is',pi
end if 
call mpi_finalize(ierr)
end program pivalue
