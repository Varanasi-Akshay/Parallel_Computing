!****************************************************************
!***
!*** This file belongs with the course
!*** Introduction to Scientific Programming in C++/Fortran2003
!*** copyright 2017 Victor Eijkhout eijkhout@tacc.utexas.edu
!***
!*** type.F90 : fortran types, definition, use, parameter passing
!***
!****************************************************************

Program size

  !!codesnippet ftype
  type point
     real :: x,y
  end type point

  type(point) :: p1,p2
  p1 = point(2.5, 3.7)

  p2 = p1
  print *,p2%x,p2%y
  !!codesnippet end
  
contains
  !!codesnippet ftypepass
  real(4) function length(p)
    implicit none
    type(point),intent(in) :: p
    length = sqrt( p%x**2 + p%y )
  end function length
  !!codesnippet end
  
end Program size
