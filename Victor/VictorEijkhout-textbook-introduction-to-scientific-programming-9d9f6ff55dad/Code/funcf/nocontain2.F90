!****************************************************************
!***
!*** This file belongs with the course
!*** Introduction to Scientific Programming in C++/Fortran2003
!*** copyright 2017 Victor Eijkhout eijkhout@tacc.utexas.edu
!***
!*** nocontains.F90 : the dangers of not using CONTAINS
!***
!****************************************************************

!!codesnippet nocontaintype
Program ContainsScope
  implicit none
  integer :: i=5
  call DoWhat(i)
end Program ContainsScope

subroutine DoWhat(x)
  implicit none
  real :: x
  print *,x
end subroutine DoWhat
!!codesnippet end
