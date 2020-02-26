!****************************************************************
!***
!*** This file belongs with the course
!*** Introduction to Scientific Programming in C++/Fortran2003
!*** copyright 2017 Victor Eijkhout eijkhout@tacc.utexas.edu
!***
!*** nocontain.F90 : the dangers of not using CONTAINS
!***
!****************************************************************

!!codesnippet nocontain
Program ContainsScope
  implicit none
  call DoWhat()
end Program ContainsScope

subroutine DoWhat(i)
  implicit none
  integer :: i
  i = 5
end subroutine DoWhat
!!codesnippet end
