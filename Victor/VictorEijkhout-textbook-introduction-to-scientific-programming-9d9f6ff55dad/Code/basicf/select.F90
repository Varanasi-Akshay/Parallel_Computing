!****************************************************************
!***
!*** This file belongs with the course
!*** Introduction to Scientific Programming in C++/Fortran2003
!*** copyright 2017 Victor Eijkhout eijkhout@tacc.utexas.edu
!***
!*** select.F90 : the Fortran SELECT statement
!***
!****************************************************************

Program SomeProgram
  implicit none

  integer :: i = 5

  !!codesnippet casef
  Select Case (i)
Case (:-1)
   print *,"Negative"
Case (5)
   print *,"Five!"
Case (0)
   print *,"Zero."
Case (1:4,6:) ! can not have (1:)
   print *,"Positive"
end Select
!!codesnippet end

End Program SomeProgram


