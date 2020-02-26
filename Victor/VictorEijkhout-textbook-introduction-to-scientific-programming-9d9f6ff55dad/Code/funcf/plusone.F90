!****************************************************************
!***
!*** This file belongs with the course
!*** Introduction to Scientific Programming in C++/Fortran2003
!*** copyright 2017 Victor Eijkhout eijkhout@tacc.utexas.edu
!***
!*** plusone.F90 : function with return type
!***
!****************************************************************

!!codesnippet fplusone
program plussing
  implicit none
  integer :: i
  i = plusone(5)
  print *,i
contains
  integer function plusone(invalue)
    implicit none
    integer,intent(in) :: invalue
    plusone = invalue+1
  end function plusone
end program plussing
!!codesnippet end
