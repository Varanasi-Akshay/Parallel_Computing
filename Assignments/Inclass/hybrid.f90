program main
use mpi
use omp_lib
implicit none
integer :: rank,size,ierr,i,count=0,id

call mpi_init(ierr)
call mpi_comm_rank(MPI_COMM_WORLD,rank,ierr)
call mpi_comm_size(MPI_COMM_WORLD,size,ierr)
call omp_set_num_threads(4)



!print *,'Hi this is',rank

!$omp parallel do reduction(+:count)
 do i=1,10
    id=omp_get_thread_num()
    print *,'Hi this is',rank,id,i
    count=count+2
 end do
!!$ omp end parallel

call mpi_barrier(MPI_COMM_WORLD,ierr)
if(rank==0) then
 !  !$ omp parallel
!     id=omp_get_thread_num()
     print *, count,'and the '
  ! !$ omp end parallel
  
end if

call mpi_barrier(MPI_COMM_WORLD,ierr)



call mpi_finalize(ierr)


end program main 
