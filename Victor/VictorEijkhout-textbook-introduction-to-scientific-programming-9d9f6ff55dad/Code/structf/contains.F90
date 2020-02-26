!****************************************************************
!***
!*** This file belongs with the course
!*** Introduction to Scientific Programming in C++/Fortran2003
!*** copyright 2017 Victor Eijkhout eijkhout@tacc.utexas.edu
!***
!*** contains.F90 : putting a function in the CONTAINS block
!***
!****************************************************************

!!codesnippet programcontains
Program size

  integer :: i=2,j
  j = twice(i)
  print *,j
contains
  integer function twice(i)
    integer,intent(in) :: i
    twice  = 2*i
  end function twice
end Program size
!!codesnippet end
