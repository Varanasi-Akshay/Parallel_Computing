!****************************************************************
!***
!*** This file belongs with the course
!*** Introduction to Scientific Programming in C++/Fortran2003
!*** copyright 2017 Victor Eijkhout eijkhout@tacc.utexas.edu
!***
!*** looppos.F90 : exercise for function with intent out
!***
!****************************************************************

!!codesnippet flooppos
program looppos
  implicit none
  real(4) :: userinput
  do while (pos_input(userinput))
     print &
      '("Positive input:",f7.3)',&
      userinput
  end do
  print &
      '("Negative input:",f7.3)',&
      userinput
  !!codesnippet end
contains
  logical function pos_input(usernumber)
    implicit none
    real(4),intent(out) :: usernumber
    read *,usernumber
    pos_input = usernumber>0
  end function pos_input
!!codesnippet flooppos
end program looppos
  !!codesnippet end
