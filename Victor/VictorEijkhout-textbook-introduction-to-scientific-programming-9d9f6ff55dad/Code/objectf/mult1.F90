!!codesnippet fmult1prog
Module multmod
  
  type Scalar
     real(4) :: value
   contains
     procedure,public :: print
     procedure,public :: scaled
  end type Scalar

contains ! methods
  !!codesnippet end

  !!codesnippet fmult1method
  subroutine print(me)
    implicit none
    class(Scalar) :: me
    print '("The value is",f7.3)',me%value
  end subroutine print
  function scaled(me,factor)
    implicit none
    class(Scalar) :: me
    real(4) :: scaled,factor
    scaled = me%value * factor
  end function scaled
  !!codesnippet end

!!codesnippet fmult1prog
end Module multmod

Program Multiply
  use multmod
  implicit none

  type(Scalar) :: x
  real(4) :: y
  x = Scalar(-3.14)
  call x%print()
  y = x%scaled(2.)
  print '(f7.3)',y

end Program Multiply
!!codesnippet end
