!****************************************************************
!***
!*** This file belongs with the course
!*** Introduction to Scientific Programming in C++/Fortran2003
!*** copyright 2017/8 Victor Eijkhout eijkhout@tacc.utexas.edu
!***
!*** shape.F90 : array reshaping
!***
!****************************************************************

Program ArrayPrint
  implicit none
  integer,parameter :: M=4,N=5
  integer :: i,j,count=1

  real(4),dimension(M,N) :: rect

  !!codesnippet printarray
  do i=1,M
     do j=1,N
        rect(i,j) = count
        count = count+1
     end do
  end do
  print *,rect
  !!codesnippet end

End Program ArrayPrint
