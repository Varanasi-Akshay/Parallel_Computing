program gather
use MPI
! Build matrix A from column vectors v; 4 processors, A=4x4.
! MAP: A = [v0,v1,v2,v3] vi = column vector from process I.
!
  integer,parameter :: N=4,p=8
  real*8 :: a(N,N),v(N)
  !include 'mpif.h’
  call mpi_init(ierr)
  call mpi_comm_rank(MPI_COMM_WORLD,mype,ierr)
  call mpi_comm_size(MPI_COMM_WORLD,npes,ierr)
  if(npes.ne.N) stop
  ! Vector Syntax (each element of v assigned mype)
  v=mype+5
  call mpi_gather(v,N,MPI_REAL8,a,N,MPI_REAL8, 0,MPI_COMM_WORLD,ierr)
  if(mype.eq.0) write(6,'(4f5.0)') ((a(i,j),j=1,N),i=1,4)
  call mpi_finalize(ierr)
  print *,p
end program gather
