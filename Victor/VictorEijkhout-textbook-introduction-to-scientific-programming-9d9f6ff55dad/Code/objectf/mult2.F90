Module multmod
  
  type Scalar
     real(4) :: value
   contains
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
  end type Scalar
  
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
