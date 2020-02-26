!****************************************************************
!***
!*** This file belongs with the course
!*** Introduction to Scientific Programming in C++/Fortran2003
!*** copyright 2017 Victor Eijkhout eijkhout@tacc.utexas.edu
!***
!*** arraypass1d.F90 : passing a 1d array
!***
!****************************************************************

Module ArrayFunction
contains

  !!codesnippet fpass1d
  real(8) function arraysum(x)
    implicit none
    real(8),intent(in),dimension(:) :: x
  !!codesnippet end
    real(8) :: tmp
    integer i

    tmp = 0.
  !!codesnippet fpass1d
    do i=1,size(x)
       tmp = tmp+x(i)
    end do
  !!codesnippet end
    arraysum = tmp
  end function arraysum

End Module ArrayFunction

#define N 100
!!codesnippet fpass1d
Program ArrayComputations1D
    use ArrayFunction
    implicit none

    real(8),dimension(:) :: x(N)
  !!codesnippet end
    real(8),dimension(:) :: y(0:N-1)
    integer :: i

    do i=1,N
       x(i) = i
       y(i-1) = i
    end do
!!codesnippet fpass1d
    print *,"Sum of one-based array:",arraysum(x)
!!codesnippet end
    print *,"Sum of zero-based array:",arraysum(y)

End Program ArrayComputations1D



