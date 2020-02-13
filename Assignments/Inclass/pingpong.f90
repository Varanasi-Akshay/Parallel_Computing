program main
use mpi
implicit none

integer :: comm=MPI_COMM_WORLD,n,i,j
integer ::status(MPI_STATUS_SIZE),nranks,rank,ierr,irec=-1


! ping - data to send, pong- recv
integer :: ping,pong

! Left proc, center proc, right proc
integer :: left=0,center=0,right=0

print *,"Hi"
call MPI_INIT(ierr)
call MPI_COMM_SIZE(comm, nranks,ierr)
call MPI_COMM_RANK(comm, rank, ierr)


! Send right algorithm
left=rank-1
center=rank
right=rank+1

if(rank==0) then
   ping=center+1 ! sending this to right proc
   call mpi_send(ping,1,MPI_int,right,right,comm,ierr)
   print *,"ping from rank",center,"to",right,"with value",ping
   call mpi_recv(pong,1,MPI_int,nranks-1,center,comm,status,ierr)
   print *,"pong from rank",nranks-1,"to",center,"with value",pong
   
else if(rank==nranks-1) then
   
   call mpi_recv(pong,1,MPI_int,left,center,comm,status,ierr)
   print *,"pong from rank",left,"to",center,"with value",pong
   ping=pong+1
   call mpi_send(ping,1,MPI_int,0,0,comm,ierr)
   print *,"ping from rank",center,"to",0,"with value",ping
   
else
   call mpi_recv(pong,1,MPI_int,left,center,comm,status,ierr)
   print *,"pong from rank",left,"to",center,"with value",pong
   ping=pong+1
   call mpi_send(ping,1,MPI_int,right,right,comm,ierr)
   print *,"ping from rank",center,"to",right,"with value",ping

end if






call MPI_FINALIZE(ierr);

end program main
