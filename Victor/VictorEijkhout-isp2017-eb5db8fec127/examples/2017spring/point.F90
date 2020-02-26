Module PointClass
  Type,public :: Point
     real(8) :: x,y
   contains
     procedure, public :: set
     procedure, public :: distance
  End type Point
  
private
contains

  subroutine set(p,xu,yu)
    implicit none
    class(point) :: p
    real(8),intent(in) :: xu,yu
    p%x = xu; p%y = yu
  end subroutine set

End Module PointClass

Program PointTest
  use PointClass
  implicit none
  type(Point) :: p1,p2

  call p1%set(1.d0,1.d0)
  call p2%set(4.d0,5.d0)

  print *,"Distance:",p1%distance(p2)

End Program PointTest
