program main
use mpi
implicit none
CHARACTER(LEN=MPI_MAX_PROCESSOR_NAME):: name
INTEGEr :: resultlen
INTEGER :: ierror,a=3
real :: a1=0.6
integer :: comm=MPI_COMM_WORLD,n,i,j
integer ::status(MPI_STATUS_SIZE),nranks,rank,ierr,irec=-1
print *,"Hi"
call MPI_INIT(ierr)
call MPI_COMM_SIZE(comm, nranks,ierr)
call MPI_COMM_RANK(comm, rank, ierr)
 if(rank==0) then
     call MPI_get_processor_name(name,resultlen,ierror)
     print *,name,resultlen
     call MPI_SEND(a , 1,MPI_REAL, 1,9, comm,ierr)
 endif    
 if(rank==1)&
      call MPI_RECV( irec, 1,MPI_INTEGER, 0,9, comm,status,ierr)

call MPI_FINALIZE(ierr);

print*,"iam=",rank,"received=",irec

end program main
