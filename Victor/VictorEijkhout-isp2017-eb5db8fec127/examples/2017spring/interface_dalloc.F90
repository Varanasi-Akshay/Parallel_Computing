program ex9
implicit none
interface build
subroutine build1(n)
   integer :: n
end subroutine
subroutine build2(n, m)
   integer :: n,m
end subroutine

end interface

integer :: n=3, m=5

call build(n)
call build(n,m)

end program

subroutine build1(n)
   implicit none
   integer :: n, ierror
   real, dimension(:), allocatable :: x
   allocate(x(n), stat = ierror)
   if (ierror /= 0) stop 'allocation error'
   call random_number(x)
   print *, x
end subroutine

subroutine build2(n, m)
   implicit none
   integer :: n, m, ierror
   real, dimension(:,:), allocatable :: x
   allocate(x(n,m), stat = ierror)
   if (ierror /= 0) stop 'allocation error'
   call random_number(x)
   print *,x
end subroutine
