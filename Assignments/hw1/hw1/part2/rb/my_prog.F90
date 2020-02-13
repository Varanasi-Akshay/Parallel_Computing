program red_black
use omp_lib

integer,parameter ::  KR8 = selected_real_kind(13)
integer,parameter ::  N=30000000
integer           ::  i, niter=0, nt=1

real(KR8)         ::  a(N), error = 1.0d0, sum

real(KR8),external::  gtod_timer                 !timer function
real(KR8)         ::  t0, t1                     !timer vars

integer,  external::  f90_setaffinity            !affinity function
integer           ::  it,itd2,icore,ierr         !affinity vars

#ifdef _OPENMP
!$omp parallel
   nt = omp_get_num_threads(); if(nt<1) print*,"fork warm up"
!$omp end parallel
#endif

   do i=1,N-1,2; a(i)   = 0.0; a(i+1) = 1.0d0; end do

   t0 = gtod_timer();

   do while (error .ge. 1.0d0)

      do i=2, N,   2;  a(i) = (a(i) + a(i-1)) / 2.0; end do
      do i=1, N-1, 2;  a(i) = (a(i) + a(i+1)) / 2.0; end do

      error=0.0d0; niter = niter+1

      do i=1,N-1; error=error + abs(a(i)-a(i+1)); end do

   end do

   t1 = gtod_timer();
   time = real(t1 - t0)

   write(*,'(f12.4)') time

end program red_black
